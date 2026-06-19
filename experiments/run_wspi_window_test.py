r"""
WSPI Window-Size Test (parallel, single-folder)
================================================
Fair test of whether the analysis window size matters at fine granularity.
Runs the SAME final WSPI (mu_L * exp(R - WE)) with different DTCWT analysis
windows, all on the SAME scenario, so we can see how RSI / ΔRank / Spearman
respond to window length — instead of tuning the window per scenario.

Variants (window in slots; min_obs = window/2):
    WSPI_w16, WSPI_w32, WSPI_w64 (the current default), WSPI_w128

Intended mainly for the 5-minute Taxi scenario, where the default w64 showed
high rank distortion and low Spearman:
    python experiments\run_wspi_window_test.py yellow_taxi --cores 4 ^
        --data-path data\datasets\<taxi_5min>.csv

But works on any dataset/granularity.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.parallel_engine import run_methods_parallel

WINDOWS = [16, 32, 64, 128]


def build_specs():
    specs = {}
    for w in WINDOWS:
        specs[f'WSPI_w{w}'] = dict(
            alpha=1.0, beta=1.0, use_R=True, use_WE=True, use_dtcwt=True,
            _window_slots=w, _min_obs=max(8, w // 2),
        )
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num-items', type=int, default=None)
    ap.add_argument('--start-date', default=None)
    ap.add_argument('--end-date', default=None)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--item-selection', default='top')
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    specs = build_specs()
    run_methods_parallel(
        args.dataset, list(specs.keys()),
        cores=args.cores, tag='windowtest', specs=specs,
        data_path=args.data_path, num_items=args.num_items,
        start_date=args.start_date, end_date=args.end_date,
        prediction_horizon=args.horizon,
        item_selection=args.item_selection, resume=args.resume,
    )


if __name__ == '__main__':
    main()
