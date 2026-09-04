# -*- coding: utf-8 -*-
"""
Prepare a comparison folder from a COMPLETED run — without re-running anything
==============================================================================
Clones the cached per-method results (protocol/ + _status.json + metadata/)
of an existing run folder into a NEW folder, so you can:

  1. keep the original thesis results 100% untouched, and
  2. `--resume` the clone with old + new methods: every cached method is
     skipped instantly ("done" marker + protocol file present) and ONLY the
     new methods are actually computed. The merged summary CSV in the clone
     then contains cached + new methods side by side.

Usage:
    python experiments\\prepare_compare_folder.py results\\youtube\\main_20260601_120000
    python experiments\\prepare_compare_folder.py results\\yellow_taxi\\main_... --tag predcmp

Then (example — only ARIMA/Holt/... will actually run):
    python experiments\\run_popularity_assessment.py youtube --cores 4 ^
        --resume results\\youtube\\predcmp_<ts> ^
        --methods WSPI AF ARIMA ARYW Holt Persistence

IMPORTANT: pass the SAME --data-path / --window-size / --horizon / --k-list
as the original run, so the new methods see identical windows and items.
Check metadata/run_metadata.json inside the clone for the original settings.

Author: Sajjad (with Claude)
Date: August 2026
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description='Clone cached results of a completed run into a new '
                    'resume-able comparison folder (originals stay untouched).')
    parser.add_argument('source', type=str,
                        help='Path to the completed run folder '
                             '(e.g. results/youtube/main_20260601_120000)')
    parser.add_argument('--tag', type=str, default='predcmp',
                        help='Name prefix of the new folder (default: predcmp)')
    args = parser.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(f'ERROR: source folder not found: {src}')

    proto = src / 'protocol'
    status = src / '_status.json'
    if not proto.exists() or not status.exists():
        sys.exit(f'ERROR: {src} does not look like a run folder '
                 f'(missing protocol/ or _status.json)')

    ts = time.strftime('%Y%m%d_%H%M%S')
    dst = src.parent / f'{args.tag}_{ts}'
    (dst / 'logs').mkdir(parents=True, exist_ok=False)

    # 1. per-method protocol files (this is what marks a method "complete")
    shutil.copytree(proto, dst / 'protocol')

    # 2. done/failed markers
    shutil.copy2(status, dst / '_status.json')

    # 3. metadata for reference (original config: windows, items, k_list, ...)
    meta = src / 'metadata'
    if meta.exists():
        shutil.copytree(meta, dst / 'metadata')

    # Report what is cached and ready to be skipped
    try:
        done = [m for m, s in json.loads(
            status.read_text(encoding='utf-8')).items() if s == 'done']
    except Exception:
        done = []

    dataset = src.parent.name
    print('=' * 72)
    print(f'COMPARISON FOLDER READY: {dst}')
    print(f'  cached (will be SKIPPED on resume): {done}')
    print('=' * 72)
    print('Next step — run ONLY the new methods and merge with the cache:')
    print()
    print(f'  python experiments\\run_popularity_assessment.py {dataset} '
          f'--cores 4 ^')
    print(f'      --resume "{dst.relative_to(ROOT)}" ^')
    print(f'      --methods WSPI AF ARIMA ARYW Holt Persistence')
    print()
    print('(add the same --data-path / --horizon / --k-list as the original '
          'run; see metadata/run_metadata.json)')
    print('Summary lands in: ' + str(
        (dst / 'comparison' / 'main_summary.csv').relative_to(ROOT)))


if __name__ == '__main__':
    main()
