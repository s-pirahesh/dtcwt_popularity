"""
Generate Figure 8 — WSPI performance versus temporal granularity on NYC Yellow
Taxi (Hourly -> 30min -> 5min).

This is a companion to ``generate_paper_figures.py``.  It does NOT modify
``cross_dataset_visualizer.py``; instead it reuses that module's data pipeline
(``CrossDatasetVisualizer``) and its exact style constants (method colours,
metric labels, fonts, grid, DPI) so the output is visually consistent with the
group_by="dataset" figures (figs 2-5).

Unlike figs 2-5 (grouped bars over 9 methods x 4 scenarios), Fig 8 is a
two-panel line-trend chart:
    (a) RSI@10   rising  as granularity gets finer   (higher is better)
    (b) dRank    falling as granularity gets finer   (lower  is better)
Three lines per panel: WSPI, DTCWT+AF, and a "Best traditional" reference
(best baseline per granularity, excluding the degenerate PFRF).

Usage (Windows PowerShell) — same four run dirs you already use:
    python scripts\\generate_granularity_trend.py `
        --youtube         "results\\youtube\\main_20260612_140555" `
        --yellow-taxi-h   "results\\yellow_taxi\\main_20260612_140633_hourly" `
        --yellow-taxi-30m "results\\yellow_taxi\\main_20260620_092517_30min" `
        --yellow-taxi-5m  "results\\yellow_taxi\\main_20260620_170044_5min" `
        --out             "paper_figures\\dataset_view_new" `
        --show

(--youtube is accepted for a drop-in identical invocation but is not used by
this figure.)

Output
------
    <out>/fig8_granularity_trend.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--show" in sys.argv:
    os.environ.setdefault("CROSS_VIZ_BACKEND", "TkAgg")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import numpy as np

# Reuse the visualizer's data pipeline AND its style constants so the new
# figure is the same "family" as figs 2-5.
from evaluation.cross_dataset_visualizer import (  # type: ignore
    CrossDatasetVisualizer,
    METHOD_COLORS,
    METRIC_LABELS,
    BASELINE_METHODS,
    WAVELET_TICK_COLOR,
    PROPOSED_EDGE_COL,
    GRID_ALPHA,
    FIG_DPI,
)
import matplotlib.pyplot as plt


# Three taxi granularities, in order, with short x-axis labels.
TAXI_DATASETS = [
    ("NYC Yellow Taxi Hourly", "Hourly"),
    ("NYC Yellow Taxi 30m",    "30-min"),
    ("NYC Yellow Taxi 5m",     "5-min"),
]

# Best-traditional reference excludes PFRF (degenerate: NDCG@10 ~ 0.2-0.6).
TRADITIONAL = [m for m in BASELINE_METHODS if m != "PFRF"]

# Per-method line style. Colours come straight from METHOD_COLORS so WSPI and
# DTCWT+AF match their bars in figs 2-5. WSPI "pops" (solid, thick, dark marker
# edge) the same way it does in the bar charts.
LINE_STYLE = {
    "WSPI":      dict(color=METHOD_COLORS["WSPI"],     ls="-",  lw=2.6, marker="o",
                      ms=7.5, markeredgecolor=PROPOSED_EDGE_COL, markeredgewidth=0.9, zorder=6),
    "DTCWT+AF":  dict(color=METHOD_COLORS["DTCWT+AF"], ls="--", lw=2.0, marker="s",
                      ms=6.5, markeredgecolor=PROPOSED_EDGE_COL, markeredgewidth=0.7, zorder=5),
    "Best traditional": dict(color="#888888", ls=":", lw=1.9, marker="^",
                             ms=6.5, zorder=4),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Fig 8 (granularity trend) in the dataset-view theme.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--youtube",         dest="youtube",       required=False,
                   help="(accepted for a drop-in identical call; not used here)")
    p.add_argument("--yellow-taxi-h",   dest="yellow_taxi_h",  required=True)
    p.add_argument("--yellow-taxi-30m", dest="yellow_taxi_30m", required=True)
    p.add_argument("--yellow-taxi-5m",  dest="yellow_taxi_5m",  required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _best_traditional(viz: CrossDatasetVisualizer, dataset: str, metric: str) -> float:
    vals = [viz._get_value(dataset, m, metric) for m in TRADITIONAL]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return float("nan")
    # RSI: higher is better -> max; dRank: lower is better -> min.
    return max(vals) if metric == "rsi_10" else min(vals)


def _series(viz, metric):
    """Return (wspi, dtcwt, best_trad) lists across the three granularities."""
    wspi, dtcwt, best = [], [], []
    for ds, _short in TAXI_DATASETS:
        wspi.append(viz._get_value(ds, "WSPI", metric))
        dtcwt.append(viz._get_value(ds, "DTCWT+AF", metric))
        best.append(_best_traditional(viz, ds, metric))
    return wspi, dtcwt, best


def _draw_panel(ax, x, xticklabels, series_map, ylabel, title, annotate_fmt,
                annotate_below):
    ax.set_facecolor("#EAEAF2")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=GRID_ALPHA, linewidth=1.0, color="white")

    for name in ("WSPI", "DTCWT+AF", "Best traditional"):
        ax.plot(x, series_map[name], label=name, **LINE_STYLE[name])

    # Annotate the WSPI points (it's "ours").
    dy = -16 if annotate_below else 9
    for xi, yi in zip(x, series_map["WSPI"]):
        ax.annotate(annotate_fmt.format(yi), (xi, yi),
                    textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5,
                    color=METHOD_COLORS["WSPI"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(x[0] - 0.35, x[-1] + 0.35)


def main() -> int:
    args = parse_args()

    sources = {
        "NYC Yellow Taxi Hourly": Path(args.yellow_taxi_h),
        "NYC Yellow Taxi 30m":    Path(args.yellow_taxi_30m),
        "NYC Yellow Taxi 5m":     Path(args.yellow_taxi_5m),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = CrossDatasetVisualizer(sources, output_dir=out_dir)

    x = np.arange(len(TAXI_DATASETS))
    xticklabels = [s for _ds, s in TAXI_DATASETS]

    rsi_w, rsi_d, rsi_b = _series(viz, "rsi_10")
    dr_w,  dr_d,  dr_b  = _series(viz, "delta_rank")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    _draw_panel(
        ax1, x, xticklabels,
        {"WSPI": rsi_w, "DTCWT+AF": rsi_d, "Best traditional": rsi_b},
        METRIC_LABELS["rsi_10"], "(a) Temporal stability",
        annotate_fmt="{:.3f}", annotate_below=False,
    )
    ax1.set_ylim(0, max(rsi_w + rsi_d) * 1.18)

    _draw_panel(
        ax2, x, xticklabels,
        {"WSPI": dr_w, "DTCWT+AF": dr_d, "Best traditional": dr_b},
        METRIC_LABELS["delta_rank"], "(b) Noise robustness",
        annotate_fmt="{:.2f}", annotate_below=True,
    )
    ax2.set_ylim(0, max(dr_d + dr_b) * 1.15)

    # One shared legend below the figure, proposed methods bold + coloured
    # (same treatment as the bar-chart legends).
    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, frameon=False, fontsize=9, handlelength=2.2, columnspacing=2.0,
    )
    for text in legend.get_texts():
        if text.get_text() in ("WSPI", "DTCWT+AF"):
            text.set_color(WAVELET_TICK_COLOR)
            text.set_fontweight("bold")

    fig.suptitle("NYC Yellow Taxi — effect of temporal granularity", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    savepath = out_dir / "fig8_granularity_trend.png"
    fig.savefig(savepath, dpi=FIG_DPI, bbox_inches="tight")
    if args.show and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)

    print(f"Saved: {savepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
