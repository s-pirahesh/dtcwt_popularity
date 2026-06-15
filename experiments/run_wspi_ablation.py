r"""
WSPI Ablation (parallel, single-folder) — reviewer comment #5
=============================================================
Ablates the FINAL WSPI formula  mu_L * exp(alpha*R - beta*WE)  by switching
off one structural component at a time, and by swapping DTCWT -> DWT. Each
variant runs in its own process; all land in one shared results folder.

Variants:
    WSPI        full                         mu_L * exp(R - WE)
    WSPI-noR    drop energy-ratio term       mu_L * exp(-WE)
    WSPI-noWE   drop entropy term            mu_L * exp(R)
    WSPI-DWT    DTCWT replaced by DWT (db4)  (tests shift-invariance)

Usage (Windows):
    python experiments\run_wspi_ablation.py youtube --cores 4
    python experiments\run_wspi_ablation.py yellow_taxi --cores 4 ^
        --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.parallel_engine import run_methods_parallel

VARIANTS = {
    'WSPI':      dict(alpha=1.0, beta=1.0, use_R=True,  use_WE=True,  use_dtcwt=True),
    'WSPI-noR':  dict(alpha=1.0, beta=1.0, use_R=False, use_WE=True,  use_dtcwt=True),
    'WSPI-noWE': dict(alpha=1.0, beta=1.0, use_R=True,  use_WE=False, use_dtcwt=True),
    'WSPI-DWT':  dict(alpha=1.0, beta=1.0, use_R=True,  use_WE=True,  use_dtcwt=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num-items', type=int, default=None)
    ap.add_argument('--start-date', default=None)
    ap.add_argument('--end-date', default=None)
    ap.add_argument('--window-size', type=int, default=30)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--item-selection', default='top')
    ap.add_argument('--resume', type=str, default=None, help='Folder of an interrupted run to continue')
    args = ap.parse_args()

    run_methods_parallel(
        args.dataset, list(VARIANTS.keys()),
        cores=args.cores, tag='ablation', specs=VARIANTS,
        data_path=args.data_path, num_items=args.num_items,
        start_date=args.start_date, end_date=args.end_date,
        window_size=args.window_size, prediction_horizon=args.horizon,
        item_selection=args.item_selection, resume=args.resume,
    )


if __name__ == '__main__':
    main()
