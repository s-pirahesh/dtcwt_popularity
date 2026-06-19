"""
generate_sparsity_figure.py
---------------------------
Generates Figure 8 of the WSPI paper:
"Signal sparsity versus temporal granularity on NYC Yellow Taxi"

For each temporal granularity it computes, over the full (zone x time-slot) panel:
  1) zero-cell ratio (%)            -> fraction of (zone, slot) pairs with no trips
  2) mean count per active slot     -> average trips over the non-empty cells

The two quantities together explain why WSPI (and complex-wavelet analysis in
general) degrades at very fine granularities: as the bins shrink, the per-zone
demand signal becomes sparse and small-valued, so the DTCWT trend band carries
little stable energy.

Input format (CSV or TSV), one row per (slot, item):
    timestamp,item_id,count
Column order does not matter; a header is required.

Usage
-----
    python generate_sparsity_figure.py \
        --hourly  data/taxi_1h.csv \
        --half    data/taxi_30min.csv \
        --fine    data/taxi_5min.csv \
        --out     figures/fig8_sparsity_vs_granularity.png

If no paths are given, the DEFAULT_PATHS below are used. Edit them to match
your environment (the prepared-data files produced by the data converter).
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---- edit these defaults to point at your prepared data files --------------
DEFAULT_PATHS = {
    "Hourly": "data/taxi_1h.csv",
    "30-min": "data/taxi_30min.csv",
    "5-min":  "data/taxi_5min.csv",
}
# ----------------------------------------------------------------------------


def _read_counts(path):
    """Read a (timestamp, item_id, count) table from CSV or TSV."""
    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else None  # None -> sniff
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"timestamp", "item_id", "count"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: expected columns {required}, got {set(df.columns)}")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    return df[["timestamp", "item_id", "count"]]


def sparsity_stats(df):
    """Return (zero_cell_ratio_pct, mean_count_per_active_slot) over the full panel."""
    n_slots = df["timestamp"].nunique()
    n_zones = df["item_id"].nunique()
    total_cells = n_slots * n_zones

    active = df[df["count"] > 0]
    n_active = len(active)

    zero_ratio = 100.0 * (1.0 - n_active / total_cells) if total_cells else float("nan")
    mean_active = active["count"].sum() / n_active if n_active else float("nan")
    return zero_ratio, mean_active, n_zones, n_slots


def main():
    ap = argparse.ArgumentParser(description="WSPI Figure 8 - sparsity vs granularity")
    ap.add_argument("--hourly", default=DEFAULT_PATHS["Hourly"])
    ap.add_argument("--half",   default=DEFAULT_PATHS["30-min"])
    ap.add_argument("--fine",   default=DEFAULT_PATHS["5-min"])
    ap.add_argument("--out",    default="fig8_sparsity_vs_granularity.png")
    args = ap.parse_args()

    sources = [("Hourly", args.hourly), ("30-min", args.half), ("5-min", args.fine)]

    labels, zero_ratios, mean_counts = [], [], []
    print(f"{'Granularity':>12} | {'zero-cell %':>11} | {'mean/active':>11} | "
          f"{'zones':>6} | {'slots':>7}")
    print("-" * 60)
    for label, path in sources:
        if not os.path.exists(path):
            sys.exit(f"ERROR: file not found: {path}\n"
                     f"Edit DEFAULT_PATHS or pass --hourly/--half/--fine.")
        df = _read_counts(path)
        zr, mc, nz, ns = sparsity_stats(df)
        labels.append(label); zero_ratios.append(zr); mean_counts.append(mc)
        print(f"{label:>12} | {zr:>10.1f}% | {mc:>11.2f} | {nz:>6d} | {ns:>7d}")

    # ---- plot: dual-axis grouped bars --------------------------------------
    x = range(len(labels))
    w = 0.38
    fig, ax_left = plt.subplots(figsize=(7, 4.2))
    ax_right = ax_left.twinx()

    c_zero, c_mean = "#C0504D", "#1F77B4"
    bars1 = ax_left.bar([i - w / 2 for i in x], zero_ratios, width=w,
                        color=c_zero, label="Zero-cell ratio")
    bars2 = ax_right.bar([i + w / 2 for i in x], mean_counts, width=w,
                         color=c_mean, label="Mean count / active slot")

    ax_left.set_xticks(list(x)); ax_left.set_xticklabels(labels)
    ax_left.set_xlabel("Temporal granularity")
    ax_left.set_ylabel("Zero-cell ratio", color=c_zero)
    ax_left.yaxis.set_major_formatter(PercentFormatter())
    ax_left.tick_params(axis="y", labelcolor=c_zero)
    ax_right.set_ylabel("Mean count per active slot", color=c_mean)
    ax_right.tick_params(axis="y", labelcolor=c_mean)
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, max(mean_counts) * 1.25)  # headroom so labels clear the legend

    for b, v in zip(bars1, zero_ratios):
        ax_left.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=9, color=c_zero)
    for b, v in zip(bars2, mean_counts):
        ax_right.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                      ha="center", va="bottom", fontsize=9, color=c_mean)

    ax_left.set_title("Signal sparsity vs. temporal granularity (NYC Yellow Taxi)",
                      pad=34)
    handles = [bars1, bars2]
    ax_left.legend(handles, [h.get_label() for h in handles],
                   loc="lower center", bbox_to_anchor=(0.5, 1.005),
                   ncol=2, frameon=False)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
