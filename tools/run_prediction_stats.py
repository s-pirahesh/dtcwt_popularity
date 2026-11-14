r"""
Paired tests for the prediction runs (every scenario and horizon)
=================================================================
Calls ``tools/stats_report.py`` (unchanged) on every ``<root>/run_<X>/<scenario>/
h<h>`` folder, with the settings fixed before the runs:

  references   run A: WSPI-FG, WSPI-F2, WSPI          run B: WSPI-FA
  block        b_h = max(b, h) windows, with the base block b of the scenario
               (BASE_BLOCK: the block of the stored horizon-1 tests);
               at horizon h the ground truth of h consecutive windows overlaps
  test         two-sided Wilcoxon on block means, Holm within one
               scenario x horizon x metric, alpha 0.05 (column ``verdict``)
  bootstrap    circular block bootstrap, B = 10000, seed 42

Before the tests the script checks that the paired tests are really paired:

  window check   every method of a run folder has the same window ids and the
                 same number of items in every window as WSPI
  cross-run      the WSPI file of run B equals the WSPI file of run A
                 (same scenario and horizon), cell by cell

Outputs (``--out``, default ``<root>/stats``):
  run_<X>/<scenario>/h<h>/ref_<reference>/   method_summary.csv, paired_tests.csv,
                                             metadata/stats_run.json
  window_check.csv         one row per run folder and method
  all_paired_tests.csv     every paired_tests.csv with the columns run, scenario, horizon
  all_method_summary.csv   every method_summary.csv, same extra columns
  stats_plan.json          settings and the list of reports

A report that exists is not computed again (``--force`` to recompute).  Run
folders that are missing are listed and make the exit code 1, unless
``--allow-missing`` is given (the tables then hold what exists).

    python tools\run_prediction_stats.py --root results\paper2
    python tools\run_prediction_stats.py --root results\paper2 --runs A --allow-missing

Tuning and test parts (``--part``, supplementary run C)
-------------------------------------------------------
``--part all`` (default) is the plan above, unchanged (runs A and B, every
window).  ``--part tune`` and ``--part test`` split the common windows of run A
of every scenario and horizon in time, as in the first paper (``split_ids`` of
``tools/run_param_grid.py``: the first 30 % are the tuning part), and run the
same tests on one part only.  The methods of run A are not run again: their
protocol files are read and only the windows of the part are kept (the lines
of the CSV files are copied unchanged).  Checks first: the run C folder passes
the window check, its WSPI file equals the one of run A, and SMA64 equals MA64
in every metric column (same scorer); with ``C_sens``, its WSPI file equals the
one of run A and its configuration c = 4, gamma = 0.3, phi_d = 1 equals WSPI-F2
of run A in every metric column.

  --part tune   one report per kappa: reference WSPI-FS<kappa> against the ten
                methods of run A (the family of the registered Pareto rule, with
                WSPI-FS<kappa> in place of WSPI-FG).  Then the fixed rule of
                Experiment_Design_v2: for every kappa, the number of cells
                (scenario x horizon) where WSPI-FS<kappa> against WSPI-F2 is
                ref_better in at least one main metric and ref_worse in none; the
                largest number wins, a tie goes to the larger kappa.
                Output <root>/stats_tune/: reports, kappa_cells.csv,
                kappa_selection.csv, kappa_selection.json.
  --part test   needs <root>/stats_tune/kappa_selection.json.  One report per
                reference (WSPI-FS<selected>, WSPI-F2, WSPI, GBM, GBM+W) with the
                ten methods of run A and the run C methods WSPI-S, SMA8, SMA16,
                SMA32, EWMA8 to EWMA64, GBM, GBM+W and WSPI-FS<selected>
                (SMA64 is MA64 and is left out).  With --runs C C_sens also
                sensitivity_summary.csv (mean of every main metric of the 18
                configurations on the test windows; description only).
                Output <root>/stats_test/.

  --references  only with --part test: the references of the test reports, in
                place of the five above (same 21 methods, block, B and seed).  For
                an extra reference after the test reports exist; --out must then
                name another folder, so stats_test/ stays as it is:
                --part test --runs C --references WSPI-F-YW --out results/paper2/stats_explore

  --runs B     only with --part test: the run B methods (WSPI, WSPI-FA, ARIMA)
                on the test windows of the same split (split_ids of the run A WSPI
                windows).  Reference WSPI-FA, as on every window; the Holm family is
                the three methods of run B.  Checks first: the run B folder passes
                the window check and its WSPI file equals the one of run A.  Output
                <root>/stats_test_B/ (--out may not be stats_test/):
                --part test --runs B

  --runs D     with --part tune or --part test: the window-mean level methods
                WSPI-M (control) and WSPI-H (candidate) of run D, added to the
                methods of the test report (on the tuning part without GBM and
                GBM+W, whose tuning scores are in-sample).  References:
                  tune  WSPI-H, WSPI-M, WSPI-F2, WSPI, WSPI-F-YW
                  test  WSPI-H, WSPI-M
                Same split, block, B and seed; the selected kappa of
                stats_tune/kappa_selection.json gives WSPI-FS<kappa>.  Checks first:
                the run A, C and D folders pass the window check, the WSPI files of
                runs C and D equal the one of run A, and SMA64 equals MA64.  Output
                <root>/stats_tune_D/ or <root>/stats_test_D/ (--out may not be
                stats_tune/ or stats_test/), so the earlier reports stay as they are:
                --part tune --runs D
                --part test --runs D

The learned methods are trained on the tuning windows, so they are only part of
the test report.  Both parts write window_check.csv, part_windows.csv (number
and range of the windows of each part), all_paired_tests.csv,
all_method_summary.csv and stats_plan.json.

    python tools\run_prediction_stats.py --root results\paper2 --part tune --runs C
    python tools\run_prediction_stats.py --root results\paper2 --part test --runs C C_sens
    python tools\run_prediction_stats.py --root results\paper2 --part test --runs C ^
        --references WSPI-F-YW --out results\paper2\stats_explore
    python tools\run_prediction_stats.py --root results\paper2 --part test --runs B
    python tools\run_prediction_stats.py --root results\paper2 --part tune --runs D
    python tools\run_prediction_stats.py --root results\paper2 --part test --runs D
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import stats_report                                   # noqa: E402

BASE_BLOCK = {'youtube_hourly': 24, 'taxi_hourly': 168, 'taxi_30min': 336,
              'taxi_5min': 2016, 'movielens_daily': 7, 'movielens_weekly': 4}
HORIZONS = (1, 8, 24)
RUNS = {
    'A': dict(scenarios=tuple(BASE_BLOCK), references=('WSPI-FG', 'WSPI-F2', 'WSPI'),
              methods=('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                       'Persistence', 'Holt', 'ARYW', 'MA64')),
    'B': dict(scenarios=('youtube_hourly', 'taxi_hourly'), references=('WSPI-FA',),
              methods=('WSPI', 'WSPI-FA', 'ARIMA')),
}
N_BOOT = 10000
SEED = 42

# Supplementary runs (only with --part tune / test)
FS_KAPPAS = (1.0, 1.5, 2.0, 3.0)
FS_NAMES = tuple(f'WSPI-FS{k:g}' for k in FS_KAPPAS)
MA_LENGTHS = (8, 16, 32, 64)
SENS = tuple((c, g, p) for c in (2.0, 4.0, 8.0) for g in (0.1, 0.3, 0.5) for p in (0.8, 1.0))
SENS_NAMES = tuple(f'WSPI-F2_c{c:g}_g{g:g}_p{p:g}' for c, g, p in SENS)
SENS_DEFAULT = 'WSPI-F2_c4_g0.3_p1'
SUPP_RUNS = {
    'C': dict(scenarios=tuple(BASE_BLOCK),
              methods=('WSPI', 'WSPI-S') + FS_NAMES + tuple(f'SMA{n}' for n in MA_LENGTHS)
              + tuple(f'EWMA{n}' for n in MA_LENGTHS) + ('GBM', 'GBM+W')),
    'C_sens': dict(scenarios=tuple(BASE_BLOCK), methods=('WSPI',) + SENS_NAMES),
    'D': dict(scenarios=tuple(BASE_BLOCK), methods=('WSPI', 'WSPI-M', 'WSPI-H')),
}
LEVEL_NAMES = ('WSPI-M', 'WSPI-H')
D_REFERENCES = {'tune': ('WSPI-H', 'WSPI-M', 'WSPI-F2', 'WSPI', 'WSPI-F-YW'),
                'test': ('WSPI-H', 'WSPI-M')}
TEST_FROM_C = ('WSPI-S', 'SMA8', 'SMA16', 'SMA32', 'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64',
               'GBM', 'GBM+W')
TEST_REFERENCES = ('WSPI-F2', 'WSPI', 'GBM', 'GBM+W')        # after WSPI-FS<selected>
MAIN_METRICS = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')
ID_COLS = ('window_id', 'timestamp', 'method', 'num_items')


def block_length(scenario: str, horizon: int) -> int:
    return max(BASE_BLOCK[scenario], int(horizon))


def read_protocol(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def window_check(run: str, scenario: str, horizon: int, folder: Path, methods):
    """Rows of the window check of one run folder, and whether it passed."""
    rows, ok = [], True
    pdir = folder / 'protocol'
    found = sorted(f.name[:-len('_protocol.csv')] for f in pdir.glob('*_protocol.csv'))
    ref = read_protocol(pdir / 'WSPI_protocol.csv') if 'WSPI' in found else None
    for m in sorted(set(methods) | set(found)):
        row = dict(run=run, scenario=scenario, horizon=horizon, method=m, expected=m in methods,
                   present=m in found, n_windows=0, first_window=np.nan, last_window=np.nan,
                   same_windows=False, same_num_items=False, n_nonfinite=0, status='FAIL')
        if m in found and ref is not None:
            d = read_protocol(pdir / f'{m}_protocol.csv')
            same_w = np.array_equal(d['window_id'].to_numpy(), ref['window_id'].to_numpy())
            same_n = same_w and np.array_equal(d['num_items'].to_numpy(),
                                               ref['num_items'].to_numpy())
            row.update(n_windows=len(d),
                       first_window=int(d['window_id'].min()) if len(d) else np.nan,
                       last_window=int(d['window_id'].max()) if len(d) else np.nan,
                       same_windows=bool(same_w), same_num_items=bool(same_n),
                       n_nonfinite=int(d['n_nonfinite'].sum()))
            if same_w and same_n and m in methods:
                row['status'] = 'OK'
            elif same_w and same_n:
                row['status'] = 'EXTRA'           # a method that is not part of the run
        ok &= row['status'] == 'OK'
        rows.append(row)
    return rows, ok


def cross_run_check(scenario: str, horizon: int, folder_a: Path, folder_b: Path):
    fa, fb = folder_a / 'protocol' / 'WSPI_protocol.csv', folder_b / 'protocol' / 'WSPI_protocol.csv'
    row = dict(run='A=B', scenario=scenario, horizon=horizon, method='WSPI', expected=True,
               present=fa.exists() and fb.exists(), n_windows=0, first_window=np.nan,
               last_window=np.nan, same_windows=False, same_num_items=False,
               n_nonfinite=0, status='FAIL')
    if row['present']:
        a, b = read_protocol(fa), read_protocol(fb)
        same = (a.shape == b.shape and list(a.columns) == list(b.columns))
        if same:
            for c in a.columns:
                x, y = a[c], b[c]
                if pd.api.types.is_float_dtype(x):
                    xv, yv = x.to_numpy(float), y.to_numpy(float)
                    same &= bool(((xv == yv) | (np.isnan(xv) & np.isnan(yv))).all())
                else:
                    same &= bool((x.astype(str) == y.astype(str)).all())
        row.update(n_windows=len(a), same_windows=bool(same), same_num_items=bool(same),
                   status='OK' if same else 'FAIL')
    return row


# =============================================================================
# Tuning and test parts (--part)
# =============================================================================

def part_ids(folder_a: Path):
    """(tuning ids, test ids) of one run A folder: split_ids of the windows of
    WSPI (all methods of run A share them, see the window check)."""
    from tools.run_param_grid import split_ids
    ids = read_protocol(folder_a / 'protocol' / 'WSPI_protocol.csv')['window_id'].tolist()
    return split_ids(ids)


def same_metrics(file_a: Path, file_b: Path) -> bool:
    """True if two protocol files agree in every column except the method name."""
    if not (file_a.exists() and file_b.exists()):
        return False
    a, b = read_protocol(file_a), read_protocol(file_b)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for c in a.columns:
        if c == 'method':
            continue
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not ((xv == yv) | (np.isnan(xv) & np.isnan(yv))).all():
                return False
        elif not (x.astype(str) == y.astype(str)).all():
            return False
    return True


def equal_metric_row(run, scenario, horizon, method, file_a: Path, file_b: Path):
    ok = same_metrics(file_a, file_b)
    return dict(run=run, scenario=scenario, horizon=horizon, method=method, expected=True,
                present=file_a.exists() and file_b.exists(), n_windows=0,
                first_window=np.nan, last_window=np.nan, same_windows=ok,
                same_num_items=ok, n_nonfinite=0, status='OK' if ok else 'FAIL')


def write_part(tmp: Path, sources, ids) -> Path:
    """Folder <tmp>/protocol with the lines of the given protocol files whose
    window_id is in ``ids`` (lines copied unchanged).  sources: {method: file}."""
    keep = set(int(k) for k in ids)
    pdir = tmp / 'protocol'
    pdir.mkdir(parents=True, exist_ok=True)
    for m, f in sources.items():
        with open(f, 'r', encoding='utf-8', newline='') as fh:
            lines = fh.readlines()
        head = lines[0].rstrip('\r\n').split(',')
        if head[0] != 'window_id':
            raise ValueError(f'{f}: first column is not window_id')
        out = [lines[0]] + [ln for ln in lines[1:] if ln.strip() and int(ln.split(',', 1)[0]) in keep]
        with open(pdir / f'{m}_protocol.csv', 'w', encoding='utf-8', newline='') as fh:
            fh.writelines(out)
    return tmp


def analyse_part(sources, ids, out: Path, block: int, reference: str, B: int, part: str):
    """stats_report.analyse on the windows ``ids`` of the given files."""
    with tempfile.TemporaryDirectory() as td:
        run = write_part(Path(td), sources, ids)
        stats_report.analyse(run, out, block, None, reference, B, SEED, None)
    mf = out / 'metadata' / 'stats_run.json'
    meta = json.loads(mf.read_text(encoding='utf-8'))
    meta.update(source_run=f'part {part} of the protocol files below',
                part=part, n_part_windows=len(ids),
                source_files={m: str(Path(f).as_posix()) for m, f in sources.items()})
    mf.write_text(json.dumps(meta, indent=2), encoding='utf-8')


def select_kappa(paired: pd.DataFrame):
    """The fixed rule of Experiment_Design_v2 (section X4) on the tuning
    reports.  Returns (cells, selection table, selected kappa or None)."""
    rows = []
    for kappa, ref in zip(FS_KAPPAS, FS_NAMES):
        for (s, h), d in paired[paired['reference'] == ref].groupby(['scenario', 'horizon']):
            row = dict(kappa=kappa, reference=ref, scenario=s, horizon=int(h))
            v = []
            for c in MAIN_METRICS:
                r = d[(d['method'] == 'WSPI-F2') & (d['metric'] == c)]
                row[f'verdict_{c}'] = r['verdict'].iloc[0] if len(r) == 1 else 'missing'
                v.append(row[f'verdict_{c}'])
            row['n_better'] = v.count('ref_better')
            row['n_worse'] = v.count('ref_worse')
            row['complete'] = 'missing' not in v
            row['success'] = bool(row['complete'] and row['n_better'] >= 1 and row['n_worse'] == 0)
            rows.append(row)
    cells = pd.DataFrame(rows)
    srows = []
    for kappa, ref in zip(FS_KAPPAS, FS_NAMES):
        d = cells[cells['kappa'] == kappa] if len(cells) else cells
        srows.append(dict(kappa=kappa, reference=ref, n_cells=len(d),
                          n_complete=int(d['complete'].sum()) if len(d) else 0,
                          n_success=int(d['success'].sum()) if len(d) else 0))
    sel = pd.DataFrame(srows)
    expected = len(BASE_BLOCK) * len(HORIZONS)
    complete = bool((sel['n_complete'] == expected).all())
    best = None
    if complete:
        top = sel['n_success'].max()
        best = float(sel[sel['n_success'] == top]['kappa'].max())   # tie: larger kappa
    sel['selected'] = sel['kappa'] == best if best is not None else False
    return cells, sel, best


def stack(reports, out: Path, extra_cols=('part',)):
    for name in ('paired_tests', 'method_summary'):
        parts = []
        for r in reports:
            d = pd.read_csv(out / r['folder'] / f'{name}.csv', float_precision='round_trip')
            if name == 'method_summary':
                d.insert(0, 'reference_report', r['reference'])
            d.insert(0, 'horizon', r['horizon'])
            d.insert(0, 'scenario', r['scenario'])
            for c in extra_cols[::-1]:
                d.insert(0, c, r[c])
            parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f'all_{name}.csv', index=False,
                                                       encoding='utf-8')


def main_part_b(a, root: Path) -> int:
    """--part test --runs B: the run B methods on the test windows of run A.

    The split is the one of run C (split_ids of the run A WSPI windows); the run
    B WSPI file must equal the run A one, so both runs have the same windows.
    Reference WSPI-FA, as on every window.  Run B has no free parameter, so it
    has no tuning part."""
    if a.part != 'test':
        print('--runs B works with --part test only (run B has no tuning part)')
        return 2
    if a.references is not None:
        print('--references is not used with --runs B (the reference is WSPI-FA)')
        return 2
    out = Path(a.out) if a.out else root / 'stats_test_B'
    out = out if out.is_absolute() else ROOT / out
    if out.resolve() == (root / 'stats_test').resolve():
        print('--runs B: give --out another folder (stats_test/ is kept as it is)')
        return 2
    out.mkdir(parents=True, exist_ok=True)
    spec = RUNS['B']
    t0 = time.time()
    wrows, prow, missing, failed, reports = [], [], [], [], []
    for s in spec['scenarios']:
        if a.scenarios and s not in a.scenarios:
            continue
        for h in a.horizons:
            fa = root / 'run_A' / s / f'h{h}'
            fb = root / 'run_B' / s / f'h{h}'
            tag = f'{s}/h{h}'
            gone = [str(f.relative_to(root).as_posix()) for f in (fa, fb)
                    if not (f / 'metadata' / 'protocol_v5_run.json').exists()]
            if gone:
                missing += gone
                continue
            rows, ok = window_check('B', s, h, fb, spec['methods'])
            r = cross_run_check(s, h, fa, fb)
            rows.append(r)
            ok &= r['status'] == 'OK'
            wrows += rows
            if not ok:
                failed.append(tag)
                print(f'WINDOW CHECK FAILED  {tag}  (no tests for this folder)')
                continue
            tune, test = part_ids(fa)
            prow.append(dict(scenario=s, horizon=h, n_windows=len(tune) + len(test),
                             n_tune=len(tune), first_tune=tune[0] if tune else np.nan,
                             last_tune=tune[-1] if tune else np.nan, n_test=len(test),
                             first_test=test[0] if test else np.nan,
                             last_test=test[-1] if test else np.nan, part='test'))
            b = block_length(s, h)
            src = {m: fb / 'protocol' / f'{m}_protocol.csv' for m in spec['methods']}
            for ref in spec['references']:
                o = out / s / f'h{h}' / f'ref_{ref}'
                done = (o / 'paired_tests.csv').exists() and \
                    (o / 'metadata' / 'stats_run.json').exists()
                if done and not a.force:
                    state = 'kept'
                else:
                    t1 = time.time()
                    analyse_part(src, test, o, b, ref, a.B, 'test')
                    state = f'{time.time() - t1:.1f}s'
                reports.append(dict(part='test', run='B', scenario=s, horizon=h, reference=ref,
                                    block=b, folder=str(o.relative_to(out).as_posix())))
                print(f'test run_B/{tag:<20} ref {ref:<8} block {b:<5} {state}', flush=True)

    pd.DataFrame(wrows).to_csv(out / 'window_check.csv', index=False, encoding='utf-8')
    pd.DataFrame(prow).to_csv(out / 'part_windows.csv', index=False, encoding='utf-8')
    stack(reports, out, extra_cols=('part', 'run'))
    plan = dict(root=str(root), part='test', runs=['B'], horizons=a.horizons,
                base_block=BASE_BLOCK, block_rule='max(base block, horizon)', B=a.B, seed=SEED,
                split='split_ids of tools/run_param_grid.py on the WSPI windows of run A '
                      '(first 30 % = tuning part); the run B WSPI file equals the run A one',
                references=list(spec['references']), reports=reports, missing=missing,
                window_check_failed=failed, created=time.strftime('%Y-%m-%d %H:%M:%S'),
                runtime_seconds=round(time.time() - t0, 1))
    (out / 'stats_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'\nreports: {len(reports)}; run folders missing: {len(missing)}; '
          f'window check failed: {len(failed)}; {time.time() - t0:.0f}s')
    for m in missing:
        print(f'  missing: {m}')
    print(f'Saved to {out}')
    bad = bool(failed) or (bool(missing) and not a.allow_missing) or a.B != N_BOOT
    if a.B != N_BOOT:
        print(f'NOTE: B = {a.B} is not the registered value {N_BOOT}; test run only')
    print('STATS FAILED' if bad else 'STATS OK')
    return 1 if bad else 0


def main_part_d(a, root: Path) -> int:
    """--part tune|test --runs D: WSPI-M and WSPI-H of run D with the methods of
    the test report, on one part (references D_REFERENCES[part]).

    The methods of runs A and C are not run again; their protocol files are read
    and only the windows of the part are kept, as in ``main_part``.  On the
    tuning part GBM and GBM+W are left out (their tuning scores are in-sample)."""
    part = a.part
    if a.references is not None:
        print('--references is not used with --runs D (the references are fixed)')
        return 2
    out = Path(a.out) if a.out else root / f'stats_{part}_D'
    out = out if out.is_absolute() else ROOT / out
    if out.resolve() in ((root / 'stats_tune').resolve(), (root / 'stats_test').resolve()):
        print('--runs D: give --out another folder (stats_tune/ and stats_test/ are kept '
              'as they are)')
        return 2
    sf = root / 'stats_tune' / 'kappa_selection.json'
    if not sf.exists():
        print(f'missing {sf}: run --part tune --runs C first')
        return 2
    kappa = json.loads(sf.read_text(encoding='utf-8')).get('selected_kappa')
    if kappa is None:
        print(f'{sf} has no selected kappa (incomplete tuning reports)')
        return 2
    fs_sel = f'WSPI-FS{float(kappa):g}'
    from_c = tuple(m for m in TEST_FROM_C
                   if part == 'test' or m not in ('GBM', 'GBM+W')) + (fs_sel,)
    refs = D_REFERENCES[part]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wrows, prow, missing, failed, reports = [], [], [], [], []
    for s in BASE_BLOCK:
        if a.scenarios and s not in a.scenarios:
            continue
        for h in a.horizons:
            fa = root / 'run_A' / s / f'h{h}'
            fc = root / 'run_C' / s / f'h{h}'
            fd = root / 'run_D' / s / f'h{h}'
            tag = f'{s}/h{h}'
            gone = [str(f.relative_to(root).as_posix()) for f in (fa, fc, fd)
                    if not (f / 'metadata' / 'protocol_v5_run.json').exists()]
            if gone:
                missing += gone
                continue
            rows, ok = window_check('A', s, h, fa, RUNS['A']['methods'])
            r2, ok2 = window_check('C', s, h, fc, SUPP_RUNS['C']['methods'])
            r3, ok3 = window_check('D', s, h, fd, SUPP_RUNS['D']['methods'])
            rows += r2 + r3
            ok &= ok2 and ok3
            for run, f in (('A=C', fc), ('A=D', fd)):
                r = cross_run_check(s, h, fa, f)
                r['run'] = run
                rows.append(r)
                ok &= r['status'] == 'OK'
            r = equal_metric_row('MA64=SMA64', s, h, 'SMA64', fa / 'protocol' / 'MA64_protocol.csv',
                                 fc / 'protocol' / 'SMA64_protocol.csv')
            rows.append(r)
            ok &= r['status'] == 'OK'
            wrows += rows
            if not ok:
                failed.append(tag)
                print(f'WINDOW CHECK FAILED  {tag}  (no tests for this folder)')
                continue
            tune, test = part_ids(fa)
            ids = tune if part == 'tune' else test
            prow.append(dict(scenario=s, horizon=h, n_windows=len(tune) + len(test),
                             n_tune=len(tune), first_tune=tune[0] if tune else np.nan,
                             last_tune=tune[-1] if tune else np.nan, n_test=len(test),
                             first_test=test[0] if test else np.nan,
                             last_test=test[-1] if test else np.nan, part=part))
            b = block_length(s, h)
            src = {m: fa / 'protocol' / f'{m}_protocol.csv' for m in RUNS['A']['methods']}
            for m in from_c:
                src[m] = fc / 'protocol' / f'{m}_protocol.csv'
            for m in LEVEL_NAMES:
                src[m] = fd / 'protocol' / f'{m}_protocol.csv'
            for ref in refs:
                o = out / s / f'h{h}' / f'ref_{ref}'
                done = (o / 'paired_tests.csv').exists() and \
                    (o / 'metadata' / 'stats_run.json').exists()
                if done and not a.force:
                    state = 'kept'
                else:
                    t1 = time.time()
                    analyse_part(src, ids, o, b, ref, a.B, part)
                    state = f'{time.time() - t1:.1f}s'
                reports.append(dict(part=part, run='D', scenario=s, horizon=h, reference=ref,
                                    block=b, folder=str(o.relative_to(out).as_posix())))
                print(f'{part} run_D/{tag:<20} ref {ref:<10} block {b:<5} {state}', flush=True)

    pd.DataFrame(wrows).to_csv(out / 'window_check.csv', index=False, encoding='utf-8')
    pd.DataFrame(prow).to_csv(out / 'part_windows.csv', index=False, encoding='utf-8')
    stack(reports, out, extra_cols=('part', 'run'))
    plan = dict(root=str(root), part=part, runs=['D'], horizons=a.horizons,
                base_block=BASE_BLOCK, block_rule='max(base block, horizon)', B=a.B, seed=SEED,
                split='split_ids of tools/run_param_grid.py on the WSPI windows of run A '
                      '(first 30 % = tuning part); the WSPI files of runs C and D equal '
                      'the run A one',
                selected_kappa=kappa, references=list(refs),
                methods=sorted(set(src) if reports else []),
                reports=reports, missing=missing, window_check_failed=failed,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                runtime_seconds=round(time.time() - t0, 1))
    (out / 'stats_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'\nreports: {len(reports)}; run folders missing: {len(missing)}; '
          f'window check failed: {len(failed)}; {time.time() - t0:.0f}s')
    for m in missing:
        print(f'  missing: {m}')
    print(f'Saved to {out}')
    bad = bool(failed) or (bool(missing) and not a.allow_missing) or a.B != N_BOOT
    if a.B != N_BOOT:
        print(f'NOTE: B = {a.B} is not the registered value {N_BOOT}; test run only')
    print('STATS FAILED' if bad else 'STATS OK')
    return 1 if bad else 0


def main_part(a, root: Path) -> int:
    part = a.part
    runs = list(a.runs)
    if runs == ['B']:
        return main_part_b(a, root)
    if runs == ['D']:
        return main_part_d(a, root)
    if 'C' not in runs or any(r not in ('C', 'C_sens') for r in runs):
        print('--part tune/test works on run C (and C_sens), on run D alone, or on run B '
              'alone with --part test: give --runs C [C_sens], --runs D or --runs B')
        return 2
    out = Path(a.out) if a.out else root / f'stats_{part}'
    out = out if out.is_absolute() else ROOT / out
    if a.references is not None:
        if part != 'test':
            print('--references works with --part test only')
            return 2
        if out.resolve() == (root / 'stats_test').resolve():
            print('--references: give --out another folder (stats_test/ is kept as it is)')
            return 2
    out.mkdir(parents=True, exist_ok=True)
    kappa = None
    if part == 'test':
        sf = root / 'stats_tune' / 'kappa_selection.json'
        if not sf.exists():
            print(f'missing {sf}: run --part tune first')
            return 2
        kappa = json.loads(sf.read_text(encoding='utf-8')).get('selected_kappa')
        if kappa is None:
            print(f'{sf} has no selected kappa (incomplete tuning reports)')
            return 2
        fs_sel = f'WSPI-FS{float(kappa):g}'
        refs = tuple(a.references) if a.references is not None else (fs_sel,) + TEST_REFERENCES
        family = set(RUNS['A']['methods']) | set(TEST_FROM_C) | {fs_sel}
        unknown = [r for r in refs if r not in family]
        if unknown:
            print(f'--references not among the methods of the test report: {unknown}')
            return 2
    t0 = time.time()
    wrows, prow, missing, failed, reports, sens_rows = [], [], [], [], [], []
    for s in BASE_BLOCK:
        if a.scenarios and s not in a.scenarios:
            continue
        for h in a.horizons:
            fa = root / 'run_A' / s / f'h{h}'
            fc = root / 'run_C' / s / f'h{h}'
            tag = f'{s}/h{h}'
            need = [fa, fc] + ([root / 'run_C_sens' / s / f'h{h}'] if 'C_sens' in runs else [])
            gone = [str(f.relative_to(root).as_posix()) for f in need
                    if not (f / 'metadata' / 'protocol_v5_run.json').exists()]
            if gone:
                missing += gone
                continue
            rows, ok = window_check('A', s, h, fa, RUNS['A']['methods'])
            r2, ok2 = window_check('C', s, h, fc, SUPP_RUNS['C']['methods'])
            rows += r2
            ok &= ok2
            r = cross_run_check(s, h, fa, fc)
            r['run'] = 'A=C'
            rows.append(r)
            ok &= r['status'] == 'OK'
            r = equal_metric_row('MA64=SMA64', s, h, 'SMA64', fa / 'protocol' / 'MA64_protocol.csv',
                                 fc / 'protocol' / 'SMA64_protocol.csv')
            rows.append(r)
            ok &= r['status'] == 'OK'
            if 'C_sens' in runs:
                fs = root / 'run_C_sens' / s / f'h{h}'
                r3, ok3 = window_check('C_sens', s, h, fs, SUPP_RUNS['C_sens']['methods'])
                rows += r3
                ok &= ok3
                r = cross_run_check(s, h, fa, fs)
                r['run'] = 'A=C_sens'
                rows.append(r)
                ok &= r['status'] == 'OK'
                r = equal_metric_row('F2=C_sens', s, h, SENS_DEFAULT,
                                     fa / 'protocol' / 'WSPI-F2_protocol.csv',
                                     fs / 'protocol' / f'{SENS_DEFAULT}_protocol.csv')
                rows.append(r)
                ok &= r['status'] == 'OK'
            wrows += rows
            if not ok:
                failed.append(tag)
                print(f'WINDOW CHECK FAILED  {tag}  (no tests for this folder)')
                continue
            tune, test = part_ids(fa)
            ids = tune if part == 'tune' else test
            prow.append(dict(scenario=s, horizon=h, n_windows=len(tune) + len(test),
                             n_tune=len(tune), first_tune=tune[0] if tune else np.nan,
                             last_tune=tune[-1] if tune else np.nan, n_test=len(test),
                             first_test=test[0] if test else np.nan,
                             last_test=test[-1] if test else np.nan, part=part))
            b = block_length(s, h)
            base = {m: fa / 'protocol' / f'{m}_protocol.csv' for m in RUNS['A']['methods']}
            if part == 'tune':
                jobs = [(ref, dict(base, **{ref: fc / 'protocol' / f'{ref}_protocol.csv'}))
                        for ref in FS_NAMES]
            else:
                src = dict(base)
                for m in TEST_FROM_C + (fs_sel,):
                    src[m] = fc / 'protocol' / f'{m}_protocol.csv'
                jobs = [(ref, src) for ref in refs]
            for ref, src in jobs:
                o = out / s / f'h{h}' / f'ref_{ref}'
                done = (o / 'paired_tests.csv').exists() and                     (o / 'metadata' / 'stats_run.json').exists()
                if done and not a.force:
                    state = 'kept'
                else:
                    t1 = time.time()
                    analyse_part(src, ids, o, b, ref, a.B, part)
                    state = f'{time.time() - t1:.1f}s'
                reports.append(dict(part=part, scenario=s, horizon=h, reference=ref, block=b,
                                    folder=str(o.relative_to(out).as_posix())))
                print(f'{part} {tag:<24} ref {ref:<11} block {b:<5} {state}', flush=True)
            if part == 'test' and 'C_sens' in runs:
                fs = root / 'run_C_sens' / s / f'h{h}'
                keep = set(test)
                for (c, g, p), m in zip(SENS, SENS_NAMES):
                    d = read_protocol(fs / 'protocol' / f'{m}_protocol.csv')
                    d = d[d['window_id'].isin(keep)]
                    row = dict(scenario=s, horizon=h, method=m, clamp=c, gamma=g, phi_d=p,
                               default=m == SENS_DEFAULT, n_windows=len(d))
                    for c_ in MAIN_METRICS:
                        x = d[c_].to_numpy(float)
                        x = x[np.isfinite(x)]
                        row[f'{c_}_mean'] = float(x.mean()) if len(x) else np.nan
                    sens_rows.append(row)

    pd.DataFrame(wrows).to_csv(out / 'window_check.csv', index=False, encoding='utf-8')
    pd.DataFrame(prow).to_csv(out / 'part_windows.csv', index=False, encoding='utf-8')
    stack(reports, out)
    sel_info = None
    if part == 'tune' and reports:
        paired = pd.read_csv(out / 'all_paired_tests.csv', float_precision='round_trip')
        cells, sel, best = select_kappa(paired)
        cells.to_csv(out / 'kappa_cells.csv', index=False, encoding='utf-8')
        sel.to_csv(out / 'kappa_selection.csv', index=False, encoding='utf-8')
        sel_info = dict(selected_kappa=best, selected_method=(f'WSPI-FS{best:g}' if best else None),
                        rule='most cells (scenario x horizon) with WSPI-FS<kappa> vs WSPI-F2 '
                             'ref_better in >= 1 main metric and ref_worse in none, on the '
                             'tuning part; tie -> larger kappa',
                        candidates=list(FS_KAPPAS), main_metrics=list(MAIN_METRICS),
                        n_success={str(k): int(v) for k, v in zip(sel['kappa'], sel['n_success'])},
                        complete=best is not None,
                        created=time.strftime('%Y-%m-%d %H:%M:%S'))
        (out / 'kappa_selection.json').write_text(json.dumps(sel_info, indent=2), encoding='utf-8')
        print('\nkappa  cells with Pareto success (tuning part)')
        for _, r in sel.iterrows():
            print(f"{r['kappa']:<6g} {int(r['n_success']):>3} of {int(r['n_complete'])}"
                  f"{'   <- selected' if r['selected'] else ''}")
        if best is None:
            print('NO SELECTION: tuning reports incomplete')
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(out / 'sensitivity_summary.csv', index=False,
                                       encoding='utf-8')
    plan = dict(root=str(root), part=part, runs=runs, horizons=a.horizons,
                base_block=BASE_BLOCK, block_rule='max(base block, horizon)', B=a.B, seed=SEED,
                split='split_ids of tools/run_param_grid.py on the WSPI windows of run A '
                      '(first 30 % = tuning part)',
                selected_kappa=kappa if part == 'test' else (sel_info or {}).get('selected_kappa'),
                reports=reports, missing=missing, window_check_failed=failed,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                runtime_seconds=round(time.time() - t0, 1))
    if a.references is not None:
        plan['references'] = list(refs)
    (out / 'stats_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'\nreports: {len(reports)}; run folders missing: {len(missing)}; '
          f'window check failed: {len(failed)}; {time.time() - t0:.0f}s')
    for m in missing:
        print(f'  missing: {m}')
    print(f'Saved to {out}')
    bad = bool(failed) or (bool(missing) and not a.allow_missing) or a.B != N_BOOT
    if part == 'tune' and (sel_info is None or sel_info['selected_kappa'] is None) \
            and not a.allow_missing:
        bad = True
    if a.B != N_BOOT:
        print(f'NOTE: B = {a.B} is not the registered value {N_BOOT}; test run only')
    print('STATS FAILED' if bad else 'STATS OK')
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='results/paper2')
    ap.add_argument('--out', default=None, help='default: <root>/stats')
    ap.add_argument('--runs', nargs='+', default=None,
                    choices=sorted(RUNS) + sorted(SUPP_RUNS),
                    help='default: A B (--part all) or C (--part tune / test); '
                         'B alone with --part test: run B on the test part; '
                         'D alone with --part tune / test: WSPI-M and WSPI-H')
    ap.add_argument('--part', default='all', choices=('all', 'tune', 'test'),
                    help='all windows (default, runs A and B) or the tuning / test part')
    ap.add_argument('--scenarios', nargs='+', default=None, choices=sorted(BASE_BLOCK))
    ap.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    ap.add_argument('--B', type=int, default=N_BOOT, help='bootstrap resamples (fixed: 10000)')
    ap.add_argument('--force', action='store_true', help='recompute existing reports')
    ap.add_argument('--references', nargs='+', default=None,
                    help='--part test only: references in place of the five fixed ones '
                         '(needs --out other than stats_test)')
    ap.add_argument('--allow-missing', action='store_true')
    a = ap.parse_args(argv)

    root = Path(a.root)
    root = root if root.is_absolute() else ROOT / root
    if a.runs is None:
        a.runs = ['A', 'B'] if a.part == 'all' else ['C']
    if a.part != 'all':
        return main_part(a, root)
    if any(r in SUPP_RUNS for r in a.runs):
        print('runs C, C_sens and D are only tested on a part: use --part tune or --part test')
        return 2
    out = Path(a.out) if a.out else root / 'stats'
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wrows, missing, failed, reports = [], [], [], []
    for run in a.runs:
        spec = RUNS[run]
        for s in spec['scenarios']:
            if a.scenarios and s not in a.scenarios:
                continue
            for h in a.horizons:
                folder = root / f'run_{run}' / s / f'h{h}'
                tag = f'run_{run}/{s}/h{h}'
                if not (folder / 'metadata' / 'protocol_v5_run.json').exists():
                    missing.append(tag)
                    continue
                rows, ok = window_check(run, s, h, folder, spec['methods'])
                wrows += rows
                if run == 'B' and (root / 'run_A' / s / f'h{h}' / 'protocol').is_dir():
                    r = cross_run_check(s, h, root / 'run_A' / s / f'h{h}', folder)
                    wrows.append(r)
                    ok &= r['status'] == 'OK'
                if not ok:
                    failed.append(tag)
                    print(f'WINDOW CHECK FAILED  {tag}  (no tests for this folder)')
                    continue
                b = block_length(s, h)
                for ref in spec['references']:
                    o = out / f'run_{run}' / s / f'h{h}' / f'ref_{ref}'
                    done = (o / 'paired_tests.csv').exists() and \
                        (o / 'metadata' / 'stats_run.json').exists()
                    if done and not a.force:
                        state = 'kept'
                    else:
                        t1 = time.time()
                        stats_report.analyse(folder, o, b, None, ref, a.B, SEED, None)
                        state = f'{time.time() - t1:.1f}s'
                    reports.append(dict(run=run, scenario=s, horizon=h, reference=ref,
                                        block=b, folder=str(o.relative_to(out).as_posix())))
                    print(f'{tag:<34} ref {ref:<8} block {b:<5} {state}', flush=True)

    wc = pd.DataFrame(wrows)
    wc.to_csv(out / 'window_check.csv', index=False, encoding='utf-8')
    for name in ('paired_tests', 'method_summary'):
        parts = []
        for r in reports:
            f = out / r['folder'] / f'{name}.csv'
            d = pd.read_csv(f, float_precision='round_trip')
            if name == 'method_summary':
                d.insert(0, 'reference_report', r['reference'])
            d.insert(0, 'horizon', r['horizon'])
            d.insert(0, 'scenario', r['scenario'])
            d.insert(0, 'run', r['run'])
            parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f'all_{name}.csv', index=False,
                                                       encoding='utf-8')
    plan = dict(root=str(root), runs=a.runs, horizons=a.horizons, base_block=BASE_BLOCK,
                block_rule='max(base block, horizon)', B=a.B, seed=SEED,
                references={r: list(RUNS[r]['references']) for r in a.runs},
                reports=reports, missing=missing, window_check_failed=failed,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                runtime_seconds=round(time.time() - t0, 1))
    (out / 'stats_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')

    print(f'\nreports: {len(reports)}; run folders missing: {len(missing)}; '
          f'window check failed: {len(failed)}; {time.time() - t0:.0f}s')
    for m in missing:
        print(f'  missing: {m}')
    print(f'Saved to {out}')
    bad = bool(failed) or (bool(missing) and not a.allow_missing) or a.B != N_BOOT
    if a.B != N_BOOT:
        print(f'NOTE: B = {a.B} is not the registered value {N_BOOT}; test run only')
    print('STATS FAILED' if bad else 'STATS OK')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
