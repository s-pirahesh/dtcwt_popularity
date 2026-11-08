r"""
Cut the first slots of a dataset, to time the slow scorers on a small sample
============================================================================
``tools/run_prediction.py`` with ARIMA or WSPI-FA takes hours on a full
scenario.  To measure the cost of one fit, run it on the first ``--slots``
slots of the data and read ``scorer_timing.csv``.  With 48 slots the windows
are k = 32..48 (the first ones are padded by reflection), a few thousand
item-windows.  The sample is for timing only; its results are not used.

    python tools\make_timing_sample.py --data data\datasets\youtube_hourly.csv ^
           --slots 48 --out results\paper2\smoke\timing_sample\youtube_hourly.csv
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--data', required=True)
    ap.add_argument('--slots', type=int, default=48)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    data = Path(a.data)
    data = data if data.is_absolute() else ROOT / data
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    df = pd.read_csv(data)
    keep = sorted(df['timestamp'].unique())[:a.slots]
    df = df[df['timestamp'].isin(keep)]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding='utf-8')
    print(f'{len(keep)} slots, {len(df)} rows, {df["item_id"].nunique()} items -> {out}')


if __name__ == '__main__':
    main()
