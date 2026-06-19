r"""
WSPI Old-vs-New formula comparison (parallel, single-folder)
============================================================
Direct head-to-head on the SAME scenario between:

  WSPI-NEW : mu_L * exp(R - WE)                         (final form, a=b=1)
  WSPI-OLD : mu_L * exp(clip(S_L + 0.5*R - 0.5*WE, -3, 3))   (original form,
             includes the trend-slope term and the clip)

Motivation: the slope partial-correlation diagnostic that justified dropping
S_L was run on HOURLY data. The new form's only soft spot is the 5-minute
granularity (high rank distortion, low Spearman). This run checks honestly
whether the OLD form — with its slope term — was actually better there.

Both use the 64-slot window. Run on any scenario, e.g. the 5-minute Taxi:
    python experiments\run_wspi_oldvsnew.py yellow_taxi --cores 2 ^
        --data-path data\datasets\<taxi_5min>.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.parallel_engine import run_methods_parallel

SPECS = {
    'WSPI-NEW': dict(alpha=1.0, beta=1.0, use_R=True, use_WE=True, use_dtcwt=True),
    'WSPI-OLD': dict(_kind='hybrid', alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--cores', type=int, default=2)
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--num-items', type=int, default=None)
    ap.add_argument('--start-date', default=None)
    ap.add_argument('--end-date', default=None)
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--item-selection', default='top')
    ap.add_argument('--resume', type=str, default=None)
    args = ap.parse_args()

    run_methods_parallel(
        args.dataset, list(SPECS.keys()),
        cores=args.cores, tag='oldvsnew', specs=SPECS,
        data_path=args.data_path, num_items=args.num_items,
        start_date=args.start_date, end_date=args.end_date,
        prediction_horizon=args.horizon,
        item_selection=args.item_selection, resume=args.resume,
    )


if __name__ == '__main__':
    main()
