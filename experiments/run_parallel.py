r"""
Parallel, resumable orchestrator — native, single-folder edition
================================================================
Runs the popularity-assessment methods CONCURRENTLY (one process per method,
because RSI carries sequential state across windows so windows themselves must
stay serial), and writes EVERY method into ONE shared results folder, exactly
like a normal single run.

Key properties
  * Native: this script runs each method itself via multiprocessing — it does
    NOT shell out to run_fusion_candidates.py.
  * One folder: all methods land in  results/<dataset>/parallel_<tag>_<ts>/
    (per-method files are uniquely named, so concurrent writes never collide;
    metadata is written as a full file, so it stays valid).
  * Resumable: each finished method writes a marker; a reboot only costs the
    methods that were mid-flight. Re-running the SAME command resumes, reusing
    the SAME shared folder, and re-runs only unfinished methods (clearing their
    stale partial files first).
  * Crash-isolated: each method runs in its own short-lived process.

Usage (Windows):
    python experiments\run_parallel.py youtube --cores 8
    python experiments\run_parallel.py yellow_taxi --cores 8 ^
        --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
    python experiments\run_parallel.py youtube --methods WSPI WSPI-Z2 WSPI-Z3 --cores 4
    python experiments\run_parallel.py youtube --fresh        # new folder, ignore markers
"""
import sys
# Make THIS process tolerant of a redirected (cp1252) stdout before any
# Unicode-printing import runs.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# _auto_extract imports only pandas/pathlib (not the heavy methods package).
from experiments._auto_extract import _load_protocol, _summarise_method

DEFAULT_METHODS = [
    'AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF',
    'DWT+AF', 'DTCWT+AF', 'WSPI',
    'WSPI-2', 'WSPI-Z2', 'WSPI-Z3', 'WSPI-Q2', 'WSPI-Q3', 'WSPI-Z2s', 'WSPI-Z3s',
]

# Candidate spec: name -> (mode, use_rho1, lam). 'WSPI-2' handled separately.
CAND_SPEC = {
    'WSPI-Z2':  ('z', False, 1.0),
    'WSPI-Z3':  ('z', True,  1.0),
    'WSPI-Q2':  ('quantile', False, 1.0),
    'WSPI-Q3':  ('quantile', True,  1.0),
    'WSPI-Z2s': ('z', False, 0.3),
    'WSPI-Z3s': ('z', True,  0.3),
}
FUSION_CANDS = set(CAND_SPEC) | {'WSPI-2'}


# ===========================================================================
# Worker — runs exactly ONE method into the shared folder
# ===========================================================================
def _worker(task):
    """Top-level (picklable) worker. task is a plain dict."""
    method   = task['method']
    dataset  = task['dataset']
    out_dir  = task['out_dir']
    log_path = Path(task['log_path'])

    # Redirect this worker's noisy output to its own utf-8 log file.
    try:
        fh = open(log_path, 'w', encoding='utf-8')
        sys.stdout = fh
        sys.stderr = fh
    except Exception:
        fh = None

    try:
        import experiments.run_popularity_assessment as runner
        import evaluation.method_configs as mc
        from evaluation.method_configs import MethodConfig

        # If this is a fusion candidate, register it and inject it so the
        # evaluator builds it. (Fresh process per task => no double-wrap.)
        if method in FUSION_CANDS:
            mc.METHOD_CONFIGS[method] = MethodConfig(
                name=method, window_slots=64, min_observations=32,
                description=f'Candidate: {method}')

            def _make():
                from methods.wspi_candidates import WSPI2
                if method == 'WSPI-2':
                    return WSPI2(alpha=1.0, beta=1.0, name='WSPI-2')
                from methods.wspi_fusion import WSPIFusion
                mode, use_rho1, lam = CAND_SPEC[method]
                norm = task['norm_z'] if mode == 'z' else task['norm_q']
                return WSPIFusion(mode, use_rho1, lam, normalizer=norm, name=method)

            _orig = runner.create_methods_dict

            def _patched(config):
                d = _orig(config)
                req = set(config.methods) if config.methods else None
                if req is None or method in req:
                    d[method] = _make()
                return d

            runner.create_methods_dict = _patched

        runner.run_temporal_evaluation(
            dataset_name=dataset,
            methods=[method],
            output_dir=Path(out_dir),     # <-- forces the SHARED folder
            use_timestamp=False,
            incremental=True,
            data_path=task['data_path'],
            num_items=task['num_items'],
        )
        return (method, True, None)
    except Exception:
        return (method, False, traceback.format_exc())
    finally:
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


# ===========================================================================
# Parent helpers
# ===========================================================================
def _state_dir(dataset):
    d = ROOT / 'results' / '_parallel_state' / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def _marker(dataset, method):
    return _state_dir(dataset) / f'{method}.done'.replace('+', 'PLUS').replace('/', '_')


def _logfile(dataset, method):
    return _state_dir(dataset) / (f'{method}.log'.replace('+', 'PLUS').replace('/', '_'))


def _shared_dir_file(dataset):
    return _state_dir(dataset) / 'SHARED_DIR.txt'


def _get_or_make_shared_dir(dataset, fresh):
    f = _shared_dir_file(dataset)
    if f.exists() and not fresh:
        p = Path(f.read_text(encoding='utf-8').strip())
        p.mkdir(parents=True, exist_ok=True)
        return p
    ts = time.strftime('%Y%m%d_%H%M%S')
    p = ROOT / 'results' / dataset / f'parallel_{ts}'
    p.mkdir(parents=True, exist_ok=True)
    f.write_text(str(p), encoding='utf-8')
    return p


def _is_done(dataset, method):
    mk = _marker(dataset, method)
    if not mk.exists():
        return False
    run_dir = Path(mk.read_text(encoding='utf-8').strip())
    return run_dir.exists() and _load_protocol(run_dir, method) is not None


def _clean_partial(shared_dir, method):
    """Remove a method's stale per-method files so a re-run starts clean
    (avoids duplicate appended rows in the protocol CSV)."""
    for sub, pat in [('protocol', f'{method}_protocol.*'),
                     ('detailed', f'{method}_scores.*'),
                     ('summary',  f'{method}_*.*')]:
        d = shared_dir / sub
        if d.exists():
            for fp in d.glob(pat):
                try:
                    fp.unlink()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(description='Native parallel, single-folder, resumable runner')
    ap.add_argument('dataset')
    ap.add_argument('--methods', nargs='+', default=None)
    ap.add_argument('--cores', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num-items', type=int, default=None)
    ap.add_argument('--fresh', action='store_true', help='New folder; ignore markers')
    args = ap.parse_args()

    methods = args.methods or DEFAULT_METHODS

    if args.fresh:
        for m in methods:
            _marker(args.dataset, m).unlink(missing_ok=True)
        _shared_dir_file(args.dataset).unlink(missing_ok=True)

    shared_dir = _get_or_make_shared_dir(args.dataset, args.fresh)
    todo = [m for m in methods if not _is_done(args.dataset, m)]
    done_already = [m for m in methods if m not in todo]

    # Force UTF-8 in child interpreters too (belt and suspenders).
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    print('=' * 70)
    print(f'NATIVE PARALLEL ORCHESTRATOR  dataset={args.dataset}  cores={args.cores}')
    print(f'  shared folder : {shared_dir}')
    print(f'  already done  : {len(done_already)}  {done_already}')
    print(f'  to run now    : {len(todo)}  {todo}')
    print('=' * 70)
    if not todo:
        _merge(args.dataset, methods, shared_dir)
        return

    # ---- calibrate fusion normalizers ONCE (only if needed) --------------
    norm_z = norm_q = None
    need_z = any(m in CAND_SPEC and CAND_SPEC[m][0] == 'z' for m in todo)
    need_q = any(m in CAND_SPEC and CAND_SPEC[m][0] == 'quantile' for m in todo)
    if need_z or need_q:
        print('  calibrating fusion normalizers (one pass over the data) ...')
        from methods.wspi_fusion import WSPIFusion
        import experiments.run_popularity_assessment as runner
        loader = runner.get_data_loader(args.dataset, data_path=args.data_path)
        df = loader.load_data()
        ic = 'item_id'  if 'item_id'  in df.columns else getattr(loader, 'item_col', 'item_id')
        cc = 'count'    if 'count'    in df.columns else getattr(loader, 'count_col', 'count')
        tc = 'timestamp' if 'timestamp' in df.columns else getattr(loader, 'time_col', None)
        totals = df.groupby(ic)[cc].sum().sort_values(ascending=False)
        series = []
        for it in totals.index[:2000]:
            sub = df[df[ic] == it]
            if tc and tc in sub.columns:
                sub = sub.sort_values(tc)
            series.append(sub[cc].to_numpy(dtype=float))
        if need_z:
            norm_z = WSPIFusion.calibrate(series, mode='z')
        if need_q:
            norm_q = WSPIFusion.calibrate(series, mode='quantile')
        print(f'  calibrated on {len(series)} items')

    # ---- clean stale partials for everything we are about to (re)run -----
    for m in todo:
        _clean_partial(shared_dir, m)

    tasks = [{
        'method': m, 'dataset': args.dataset, 'out_dir': str(shared_dir),
        'data_path': args.data_path, 'num_items': args.num_items,
        'log_path': str(_logfile(args.dataset, m)),
        'norm_z': norm_z if (m in CAND_SPEC and CAND_SPEC[m][0] == 'z') else None,
        'norm_q': norm_q if (m in CAND_SPEC and CAND_SPEC[m][0] == 'quantile') else None,
    } for m in todo]

    failed = []
    # Fresh process per task (isolation + no monkeypatch leakage).
    pool_kw = dict(max_workers=args.cores)
    try:
        ex = ProcessPoolExecutor(max_tasks_per_child=1, **pool_kw)  # py3.11+
    except TypeError:
        ex = ProcessPoolExecutor(**pool_kw)

    with ex:
        futs = {ex.submit(_worker, t): t['method'] for t in tasks}
        for fut in as_completed(futs):
            method = futs[fut]
            try:
                m, ok, err = fut.result()
            except Exception as e:
                m, ok, err = method, False, repr(e)
            if ok and _load_protocol(shared_dir, m) is not None:
                _marker(args.dataset, m).write_text(str(shared_dir), encoding='utf-8')
                print(f'  ✓ done   {m}')
            else:
                failed.append(m)
                print(f'  ✗ FAILED {m}  (see {_logfile(args.dataset, m).name})')
                if err:
                    print('    ' + err.strip().splitlines()[-1])

    _merge(args.dataset, methods, shared_dir)
    print('=' * 70)
    print(f'FINISHED. done={len([m for m in methods if _marker(args.dataset, m).exists()])} '
          f'failed={failed}')
    if failed:
        print('Re-run the SAME command to retry only the unfinished methods.')


def _merge(dataset, methods, shared_dir):
    """Average every done method's per-window protocol into one CSV."""
    import pandas as pd
    rows = []
    for m in methods:
        if not _marker(dataset, m).exists():
            continue
        df = _load_protocol(shared_dir, m)
        if df is None:
            continue
        row = {'method': m}
        row.update(_summarise_method(df))
        rows.append(row)
    if not rows:
        print('  merge: no completed methods yet.')
        return
    out = pd.DataFrame(rows)
    cols = ['method', 'rsi@10', 'rsi@5', 'ndcg@10', 'robustness_distortion',
            'spearman_rho', 'windows']
    ordered = [c for c in cols if c in out.columns] + \
              [c for c in out.columns if c not in cols]
    out = out[ordered]
    # Write into the shared folder's comparison/ AND results/tables/
    (shared_dir / 'comparison').mkdir(parents=True, exist_ok=True)
    out.to_csv(shared_dir / 'comparison' / 'parallel_summary.csv',
               index=False, encoding='utf-8-sig')
    tdir = ROOT / 'results' / 'tables'
    tdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out.to_csv(tdir / f'parallel_{dataset}_{ts}.csv', index=False, encoding='utf-8-sig')
    print('=' * 70)
    print(f'MERGED SUMMARY -> {shared_dir / "comparison" / "parallel_summary.csv"}')
    show = [c for c in ['method', 'rsi@10', 'ndcg@10', 'robustness_distortion']
            if c in out.columns]
    try:
        print(out[show].to_string(index=False))
    except Exception:
        pass


if __name__ == '__main__':
    main()
