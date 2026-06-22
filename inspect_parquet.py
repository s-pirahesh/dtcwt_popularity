"""
inspect_parquet.py
------------------
Diagnostic tool: reports how many TIME SLOTS / WINDOWS and which DATE RANGE a
scenario-run output actually used. Run it on the per-method *_scores.parquet
files (or any parquet/CSV produced by the evaluation) and send back the output.

Usage
-----
    # a single file:
    python inspect_parquet.py path/to/WSPI_scores.parquet

    # a whole results directory (scans *.parquet recursively):
    python inspect_parquet.py path/to/results_dir

    # also works on the prepared dataset csv:
    python inspect_parquet.py data/datasets/yellow_taxi_2025_all_5min.csv
"""
import sys, os, glob
import pandas as pd
import numpy as np


def _to_datetime(s: pd.Series) -> pd.Series:
    """Best-effort conversion of a timestamp column to datetime."""
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        m = float(np.nanmedian(s.values))
        # pick epoch unit by magnitude
        if   m > 1e17: unit = "ns"
        elif m > 1e14: unit = "us"
        elif m > 1e11: unit = "ms"
        else:          unit = "s"
        return pd.to_datetime(s, unit=unit, errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _find_time_col(df):
    for c in df.columns:
        cl = c.lower()
        if cl in ("timestamp", "time", "date", "datetime", "slot_time"):
            return c
    for c in df.columns:                       # fallback: anything time-ish
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def _infer_freq(ts_sorted):
    d = ts_sorted.drop_duplicates().sort_values().diff().dropna()
    if len(d) == 0:
        return None
    sec = d.dt.total_seconds().median()
    table = {300: "5-min", 1800: "30-min", 3600: "hourly",
             86400: "daily", 900: "15-min", 600: "10-min"}
    return table.get(int(round(sec)), f"{sec/60:.1f} min")


def inspect(path):
    print("=" * 72)
    print(f"FILE: {path}")
    try:
        df = pd.read_parquet(path) if path.lower().endswith(".parquet") \
            else pd.read_csv(path)
    except Exception as e:
        print(f"  !! could not read: {e}")
        return

    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")

    # window column, if any
    wcols = [c for c in df.columns if "window" in c.lower()]
    for c in wcols:
        print(f"  unique '{c}': {df[c].nunique():,}")

    # item column, if any
    icols = [c for c in df.columns if c.lower() in ("item_id", "item", "id")]
    for c in icols:
        print(f"  unique '{c}': {df[c].nunique():,}")

    # timestamp / slots / windows
    tcol = _find_time_col(df)
    if tcol is None:
        print("  (no timestamp-like column found)")
        return
    ts = _to_datetime(df[tcol]).dropna()
    if ts.empty:
        print(f"  (could not parse '{tcol}' as datetime)")
        return
    uniq = ts.drop_duplicates()
    span_days = (ts.max() - ts.min()).total_seconds() / 86400.0
    freq = _infer_freq(ts)
    print(f"  timestamp column: '{tcol}'")
    print(f"  UNIQUE TIME SLOTS / WINDOWS (unique timestamps): {uniq.nunique():,}")
    print(f"  date range: {ts.min()}  ->  {ts.max()}   (~{span_days:.1f} days)")
    print(f"  inferred granularity: {freq}")
    if freq in ("5-min", "30-min", "hourly", "daily"):
        per_day = {"5-min": 288, "30-min": 48, "hourly": 24, "daily": 1}[freq]
        print(f"  expected slots for ~{span_days:.0f} days at {freq}: "
              f"~{int(span_days * per_day):,}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "**", "*.parquet"),
                                 recursive=True))
        if not files:
            print(f"No .parquet files under {target}"); sys.exit(1)
        # one representative scores file is enough; show first 3
        for f in files[:3]:
            inspect(f)
        print("=" * 72)
        print(f"({len(files)} parquet files found; showed up to 3.)")
    else:
        inspect(target)


if __name__ == "__main__":
    main()
