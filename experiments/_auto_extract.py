"""
Auto-Extract Helper
===================
Shared utility used by run_ablation.py, run_sensitivity.py, and
fair_window_sensitivity.py to automatically produce a CSV summary
immediately after the standard pipeline finishes.

How it works:
    1. Before main() runs, record the current time as a fence.
    2. After main() returns, scan results/<dataset>/ for any folder whose
       mtime is AFTER the fence — that is the run folder we just created.
    3. Read every <method>_protocol.parquet/csv inside its protocol/
       sub-folder, average each metric across windows (skipna), and write
       one tidy CSV row per method.
    4. Print the final CSV path so the user knows where to find it.

The CSV uses utf-8-sig encoding so Excel opens it correctly.
"""
import sys
import time
from pathlib import Path

import pandas as pd


# Metrics to extract from each protocol file
K_VALUES = [5, 10, 20]
SCALAR_COLS = ['spearman_rho', 'kendall_tau', 'mae', 'robustness_distortion']


def _project_root():
    """Return the repository root (this file lives in experiments/)."""
    return Path(__file__).resolve().parent.parent


def find_run_dir(dataset_name, fence_time):
    """
    Find the run folder created by the most recent invocation.

    Args:
        dataset_name: e.g. 'youtube', 'yellow_taxi'
        fence_time:   epoch time recorded BEFORE the run started

    Returns:
        Path to the run folder, or None if nothing matches.
    """
    dataset_dir = _project_root() / 'results' / dataset_name
    if not dataset_dir.exists():
        return None
    candidates = [d for d in dataset_dir.iterdir()
                  if d.is_dir() and d.stat().st_mtime >= fence_time]
    if not candidates:
        return None
    # Pick the most recently modified candidate (handles edge cases)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_protocol(run_dir, method):
    """Load a method's per-window protocol DataFrame."""
    proto_dir = run_dir / 'protocol'
    for ext in ('parquet', 'csv'):
        fp = proto_dir / f'{method}_protocol.{ext}'
        if fp.exists():
            try:
                return pd.read_parquet(fp) if ext == 'parquet' else pd.read_csv(fp)
            except Exception as e:
                print(f'  WARNING: could not read {fp.name}: {e}')
                return None
    return None


def _discover_methods(run_dir):
    """Find every method that has a protocol file in this run folder."""
    proto_dir = run_dir / 'protocol'
    if not proto_dir.exists():
        return []
    found = set()
    for f in list(proto_dir.glob('*_protocol.parquet')) + \
             list(proto_dir.glob('*_protocol.csv')):
        found.add(f.stem.replace('_protocol', ''))
    return sorted(found)


def _summarise_method(df):
    """Average each metric column over all windows (skipna)."""
    out = {}
    for k in K_VALUES:
        for prefix in ('ndcg', 'rsi', 'chr'):
            col = f'{prefix}@{k}'
            if col in df.columns:
                out[col] = float(df[col].mean(skipna=True))
    for col in SCALAR_COLS:
        if col in df.columns:
            out[col] = float(df[col].mean(skipna=True))
    # Number of windows actually used
    core = 'ndcg@10' if 'ndcg@10' in df.columns else df.columns[0]
    out['windows'] = int(df[core].notna().sum())
    return out


def extract_csv(run_dir, output_csv, experiment_label=''):
    """
    Build a tidy CSV from all protocol files in run_dir.

    Args:
        run_dir:          Path to the run folder
        output_csv:       Path for the CSV to create
        experiment_label: free-form tag stored in the 'experiment' column
    """
    methods = _discover_methods(run_dir)
    if not methods:
        print(f'  No protocol files in {run_dir}; CSV not created.')
        return None

    rows = []
    for method in methods:
        df = _load_protocol(run_dir, method)
        if df is None or df.empty:
            continue
        row = _summarise_method(df)
        row['method'] = method
        row['experiment'] = experiment_label
        row['run_dir'] = run_dir.name
        rows.append(row)

    if not rows:
        return None

    df_out = pd.DataFrame(rows)
    # Reorder columns: experiment + method + metrics
    lead = ['experiment', 'method']
    trail = ['windows', 'run_dir']
    middle = [c for c in df_out.columns if c not in lead + trail]
    df_out = df_out[lead + sorted(middle) + trail]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return output_csv


def _detect_dataset_from_argv():
    """
    Look at sys.argv to figure out which dataset was requested.
    The runner takes the dataset name as the first positional argument.
    """
    valid = ['movielens', 'youtube07', 'youku', 'yellow_taxi', 'youtube']
    for arg in sys.argv[1:]:
        if arg in valid:
            return arg
    return None


def auto_extract(experiment_label):
    """
    Convenience wrapper used by the run_*.py scripts.

    Usage pattern in run_ablation.py:

        from experiments._auto_extract import fence, auto_extract
        fence_time = fence()
        ... patch configs, monkey-patch create_methods_dict ...
        runner.main()
        auto_extract('ablation')

    Args:
        experiment_label: 'ablation', 'sensitivity', 'fair_window_<N>', etc.
                          Used both in the CSV's 'experiment' column and in
                          the output filename.
    """
    # The fence is stored in a global by fence(); see below.
    fence_time = _FENCE_HOLDER.get('time', time.time() - 3600)

    dataset = _detect_dataset_from_argv()
    if dataset is None:
        print('  auto_extract: could not determine dataset from argv; skipping CSV.')
        return

    run_dir = find_run_dir(dataset, fence_time)
    if run_dir is None:
        print(f'  auto_extract: no new run folder found under '
              f'results/{dataset}/; skipping CSV.')
        return

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = (_project_root() / 'results' / 'tables'
                / f'{experiment_label}_{dataset}_{timestamp}.csv')

    print()
    print('=' * 70)
    print('AUTO-EXTRACT: producing CSV summary')
    print('=' * 70)
    print(f'  run folder : {run_dir}')
    print(f'  output CSV : {out_path}')

    saved = extract_csv(run_dir, out_path, experiment_label=experiment_label)
    if saved:
        print(f'  SUCCESS: CSV saved to {saved}')
    else:
        print('  FAILED: no rows extracted.')
    print('=' * 70)


# --- Fence (timestamp recording) -------------------------------------------
_FENCE_HOLDER = {}


def fence():
    """Record the current time so auto_extract can find new folders."""
    _FENCE_HOLDER['time'] = time.time()
    return _FENCE_HOLDER['time']
