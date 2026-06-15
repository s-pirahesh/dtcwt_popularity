r"""
WSPI Sensitivity (parallel, single-folder) — reviewer comment #4
================================================================
One-at-a-time (OAT) sweep of the two coefficients of the FINAL WSPI:
        WSPI = mu_L * exp(alpha*R - beta*WE)
  * sweep alpha in {0, 0.25, 0.5, 1, 1.5, 2}  (beta fixed = 1)
  * sweep beta  in {0, 0.25, 0.5, 1, 1.5, 2}  (alpha fixed = 1)
The point alpha=beta=1 is the proposed setting; the sweep documents that the
method is stable around it (justifies the choice without test-set tuning).

Each setting runs as its own process; all land in one shared folder. Names:
  WSPI_a0.00 ... WSPI_a2.00   and   WSPI_b0.00 ... WSPI_b2.00

Usage (Windows):
    python experiments\run_wspi_sensitivity.py youtube --cores 6
    python experiments\run_wspi_sensitivity.py yellow_taxi --cores 6 ^
        --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.parallel_engine import run_methods_parallel

GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]


def build_specs():
    specs = {}
    for a in GRID:                       # vary alpha (R), beta = 1
        specs[f'WSPI_a{a:.2f}'] = dict(alpha=a, beta=1.0,
                                       use_R=True, use_WE=True, use_dtcwt=True)
    for b in GRID:                       # vary beta (WE), alpha = 1
        specs[f'WSPI_b{b:.2f}'] = dict(alpha=1.0, beta=b,
                                       use_R=True, use_WE=True, use_dtcwt=True)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--cores', type=int, default=6)
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num-items', type=int, default=None)
    ap.add_argument('--start-date', default=None)
    ap.add_argument('--end-date', default=None)
    ap.add_argument('--window-size', type=int, default=30)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--item-selection', default='top')
    ap.add_argument('--resume', type=str, default=None, help='Folder of an interrupted run to continue')
    args = ap.parse_args()

    specs = build_specs()
    run_methods_parallel(
        args.dataset, list(specs.keys()),
        cores=args.cores, tag='sensitivity', specs=specs,
        data_path=args.data_path, num_items=args.num_items,
        start_date=args.start_date, end_date=args.end_date,
        window_size=args.window_size, prediction_horizon=args.horizon,
        item_selection=args.item_selection, resume=args.resume,
    )


if __name__ == '__main__':
    main()
