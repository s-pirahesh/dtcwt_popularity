"""
Cross-dataset comparison visualizer for the WSPI paper.

Created by: Sajjad Pirahesh, 2026-05-17

Produces one figure per metric, in either of two layout modes:

  group_by="method"   (default — used in the paper)
      X axis = methods (6 baselines | divider | 3 wavelet-based)
      Bars within each method  = datasets (colour-coded by dataset)
      Background: baselines region light grey, proposed region light pink
      WSPI bars get a bright magenta edge to pop as "ours"

  group_by="dataset"   (alternative view)
      X axis = datasets (4 scenarios)
      Bars within each dataset = methods (colour-coded by method)
      No background regions; instead, proposed-method bars (DWT+AF, DTCWT+AF,
      WSPI) get a thicker dark edge and a hatch pattern so they pop relative
      to baselines.  Legend labels for proposed methods are bold + coloured.

Accepted input formats per dataset
-----------------------------------
- A pandas DataFrame with one row per (method, window)
- Path to a single .parquet or .csv file with the same shape
- Path to a run output directory whose ``protocol/`` sub-dir holds per-method
  ``*_protocol.parquet`` files (the layout that temporal_evaluator.py writes)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import os

import matplotlib
# Backend selection.  Default is Agg (suitable for batch generation on
# headless machines).  Set CROSS_VIZ_BACKEND=TkAgg (or another interactive
# backend) before importing this module to enable plt.show() pop-ups.
_BACKEND = os.environ.get("CROSS_VIZ_BACKEND", "Agg")

def _try_use_backend(name: str) -> bool:
    """Activate `name` if it works on this machine; return True on success."""
    try:
        # Probe the backend by actually importing its module.  matplotlib.use
        # alone may succeed even if the rendering library is missing; the
        # error only surfaces later when a figure is created.
        if name.lower() in ("tkagg", "tk"):
            import tkinter  # noqa: F401
        elif name.lower() == "qt5agg":
            import PyQt5  # noqa: F401
        elif name.lower() == "qtagg":
            import PyQt6  # noqa: F401
        matplotlib.use(name, force=True)
        return True
    except (ImportError, ValueError):
        return False

if not _try_use_backend(_BACKEND):
    if _BACKEND.lower() != "agg":
        import warnings
        warnings.warn(
            f"Matplotlib backend {_BACKEND!r} not available on this machine; "
            f"falling back to 'Agg' (no interactive display).",
            RuntimeWarning,
        )
    matplotlib.use("Agg", force=True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a built-in matplotlib style that produces white gridlines on a light
# grey background (similar to seaborn's "whitegrid" theme used in the old
# visualizer.py).  No external dependency required.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Older matplotlib uses the un-versioned name
    plt.style.use("seaborn-whitegrid")


# ── Method ordering: baselines first, then wavelet-based ─────────────────────

BASELINE_METHODS: list[str] = ["AF", "CompoundPop", "EWMA", "PFRF", "RRD", "VSE"]
WAVELET_METHODS:  list[str] = ["DWT+AF", "DTCWT+AF", "WSPI"]
METHOD_ORDER: list[str] = BASELINE_METHODS + WAVELET_METHODS

# ── Dataset display order & dataset-colour palette ───────────────────────────
# Used as bar colours in group_by="method" mode and as x-axis groups in
# group_by="dataset" mode.

DATASET_ORDER: list[str] = [
    "YouTube Hourly",
    "NYC Yellow Taxi Hourly",
    "NYC Yellow Taxi 30m",
    "NYC Yellow Taxi 5m",
]

DATASET_COLORS: dict[str, str] = {
    "YouTube Hourly":         "#1f77b4",   # blue
    "NYC Yellow Taxi Hourly": "#2ca02c",   # green
    "NYC Yellow Taxi 30m":    "#ff7f0e",   # orange
    "NYC Yellow Taxi 5m":     "#9467bd",   # purple
}

# ── Method-colour palette (used as bar colours in group_by="dataset" mode) ──
# Baselines get soft, distinguishable colours; wavelet methods get strong,
# saturated colours so they stand out without needing background shading.

METHOD_COLORS: dict[str, str] = {
    # Baselines (cool, soft palette)
    "AF":          "#A6CEE3",
    "CompoundPop": "#B2DF8A",
    "EWMA":        "#FDBF6F",
    "PFRF":        "#CAB2D6",
    "RRD":         "#FB9A99",
    "VSE":         "#FFFF99",
    # Proposed methods (strong, saturated)
    "DWT+AF":      "#FF8C00",   # dark orange
    "DTCWT+AF":    "#4682B4",   # steel blue
    "WSPI":        "#C71585",   # bright magenta
}

# ── Region (background) styling for group_by="method" mode ───────────────────

BASELINE_REGION_COLOR = "#D7D7DD"   # slightly darker grey than the axes bg
BASELINE_REGION_ALPHA = 0.55
PROPOSED_REGION_COLOR = "#F7D9E8"
PROPOSED_REGION_ALPHA = 0.65
REGION_BORDER_COLOR   = "#888888"

# ── Proposed-method highlighting ─────────────────────────────────────────────

WSPI_EDGE_COLOR    = "#C71585"   # magenta edge for WSPI bars (mode 1)
WSPI_EDGE_LW       = 1.6
WAVELET_TICK_COLOR = "#7E1C9F"   # x-tick / legend label colour for proposed
PROPOSED_HATCH     = "//"        # hatch on proposed bars in mode 2
PROPOSED_EDGE_LW   = 1.2         # thicker edge on proposed bars in mode 2
PROPOSED_EDGE_COL  = "#222222"   # dark edge on proposed bars in mode 2

# ── Column name aliases ──────────────────────────────────────────────────────

_COL_ALIASES: dict[str, str] = {
    "ndcg@10":               "ndcg_10",
    "ndcg_10":               "ndcg_10",
    "rsi@10":                "rsi_10",
    "rsi_10":                "rsi_10",
    "robustness_distortion": "delta_rank",
    "delta_rank":            "delta_rank",
    "spearman_rho":          "spearman_rho",
}

_CANONICAL_COLS: list[str] = ["ndcg_10", "spearman_rho", "rsi_10", "delta_rank"]

METRIC_LABELS: dict[str, str] = {
    "ndcg_10":      "NDCG@10  (higher is better ↑)",
    "spearman_rho": "Spearman ρ  (higher is better ↑)",
    "rsi_10":       "RSI@10  (higher is better ↑)",
    "delta_rank":   "ΔRank  (lower is better ↓)",
}

METRIC_FILENAMES: dict[str, str] = {
    "ndcg_10":      "fig_ndcg10.png",
    "spearman_rho": "fig_spearman.png",
    "rsi_10":       "fig_rsi10.png",
    "delta_rank":   "fig_deltarank.png",
}

# ── Figure layout constants ──────────────────────────────────────────────────

# Mode 1: group_by="method"
M_BAR_WIDTH       = 0.18
M_GROUP_GAP_EXTRA = 0.40   # extra gap between baseline and proposed regions
M_METHOD_SPACING  = 1.0

# Mode 2: group_by="dataset"
D_BAR_WIDTH       = 0.085
D_DATASET_SPACING = 1.0

FIG_SIZE   = (12, 5.5)
FIG_DPI    = 300
GRID_ALPHA = 0.9    # near-opaque gridlines, white-on-light-grey


# ── Cross-scenario thesis-figure constants ───────────────────────────────────
# These belong to the ablation / sensitivity figures consolidated from the
# standalone script viz_additions.py.  They use their OWN palette and
# darkgrid style (their method variants are outside METHOD_COLORS /
# METHOD_ORDER above) and are deliberately independent of the
# group_by="method"/"dataset" machinery above (the two layout modes only
# apply to the per-metric bar charts).

_SCENARIO_STYLE = "seaborn-v0_8-darkgrid"
if _SCENARIO_STYLE not in plt.style.available:
    _SCENARIO_STYLE = (
        "seaborn-darkgrid" if "seaborn-darkgrid" in plt.style.available else "default"
    )

# Palette used by the scenario figures (distinct from METHOD_COLORS above).
_SCENARIO_METHOD_COLORS: dict[str, str] = {
    "AF": "#2196F3", "EWMA": "#FF9800", "RRD": "#00BCD4", "VSE": "#795548",
    "CompoundPop": "#607D8B", "PFRF": "#009688",
    "DWT+AF": "#9C27B0", "DTCWT+AF": "#F44336", "WSPI": "#E91E63",
}

# Ablation-figure ordering, labels and colours.
_ABL_ORDER: list[str] = ["WSPI", "WSPI-noR", "WSPI-noWE", "WSPI-DWT"]
_ABL_LABELS: dict[str, str] = {
    "WSPI": "WSPI (full)",
    "WSPI-noR": "w/o Energy Ratio (R)",
    "WSPI-noWE": "w/o Wavelet Entropy (WE)",
    "WSPI-DWT": "DWT instead of DTCWT",
}
_ABL_COLORS: dict[str, str] = {
    "WSPI": _SCENARIO_METHOD_COLORS["WSPI"], "WSPI-noR": "#FF9800",
    "WSPI-noWE": "#2196F3", "WSPI-DWT": "#9C27B0",
}


# ─────────────────────────────────────────────────────────────────────────────

class CrossDatasetVisualizer:
    """Aggregate per-window results from multiple datasets and plot per-metric."""

    def __init__(
        self,
        dataset_results: dict,
        output_dir: Union[Path, str],
    ) -> None:
        self.output_dir = Path(output_dir)
        self._raw: dict[str, pd.DataFrame] = {
            label: self._load_data(src)
            for label, src in dataset_results.items()
        }
        self._agg: pd.DataFrame = self._compute_aggregates()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self, source) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return self._normalise_columns(source)

        path = Path(source)

        if path.is_file():
            df = self._read_file(path)
            return self._normalise_columns(df)

        if path.is_dir():
            proto_dir  = path / "protocol"
            search_dir = proto_dir if proto_dir.is_dir() else path

            files = sorted(search_dir.glob("*.parquet"))
            if not files:
                files = sorted(search_dir.glob("*.csv"))
            if not files:
                raise FileNotFoundError(
                    f"No parquet/csv files found under {search_dir}"
                )

            frames = [self._read_file(f) for f in files]
            df = pd.concat(frames, ignore_index=True)
            return self._normalise_columns(df)

        raise FileNotFoundError(f"Source not found: {source}")

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
        return df.rename(columns=rename)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _compute_aggregates(self) -> pd.DataFrame:
        rows: list[dict] = []

        for dataset, df in self._raw.items():
            for method in METHOD_ORDER:
                if "method" not in df.columns:
                    warnings.warn(
                        f"No 'method' column found for dataset={dataset!r}; skipping."
                    )
                    break

                mdf = df[df["method"] == method]

                if mdf.empty:
                    warnings.warn(
                        f"No rows for method={method!r} in dataset={dataset!r}."
                    )
                    row: dict = {"dataset": dataset, "method": method}
                    row.update({c: np.nan for c in _CANONICAL_COLS})
                    rows.append(row)
                    continue

                row = {"dataset": dataset, "method": method}
                for col in _CANONICAL_COLS:
                    if col not in mdf.columns:
                        row[col] = np.nan
                    else:
                        if col == "spearman_rho":
                            series = mdf.loc[mdf["method"] != "LRU", col].dropna()
                        else:
                            series = mdf[col].dropna()
                        row[col] = float(series.mean()) if len(series) else np.nan

                rows.append(row)

        return pd.DataFrame(rows)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ordered_datasets(self) -> list[str]:
        present = list(self._raw.keys())
        ordered = [d for d in DATASET_ORDER if d in present]
        ordered += [d for d in present if d not in ordered]
        return ordered

    def _get_value(self, dataset: str, method: str, metric: str) -> float:
        mask = ((self._agg["dataset"] == dataset)
                & (self._agg["method"] == method))
        val = self._agg.loc[mask, metric]
        if val.empty or val.isna().all():
            return np.nan
        return float(val.iloc[0])

    # ──────────────────────────────────────────────────────────────────────────
    # Mode 1: group_by="method"
    # ──────────────────────────────────────────────────────────────────────────

    def _method_xpositions(self) -> tuple[np.ndarray, np.ndarray, float]:
        n_base = len(BASELINE_METHODS)
        n_wav  = len(WAVELET_METHODS)

        xpos_base = np.arange(n_base) * M_METHOD_SPACING
        gap_start = xpos_base[-1] + M_METHOD_SPACING + M_GROUP_GAP_EXTRA
        xpos_wav  = gap_start + np.arange(n_wav) * M_METHOD_SPACING

        divider_x = (xpos_base[-1] + xpos_wav[0]) / 2.0
        return xpos_base, xpos_wav, divider_x

    def _plot_by_method(
        self,
        metric: str,
        savepath: Union[Path, str],
        show: bool = False,
    ) -> None:
        datasets   = self._ordered_datasets()
        n_datasets = len(datasets)

        xpos_base, xpos_wav, divider_x = self._method_xpositions()
        all_xpos = np.concatenate([xpos_base, xpos_wav])

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.set_facecolor("#EAEAF2")   # light grey axes background (seaborn-like)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=GRID_ALPHA, linewidth=1.0, color="white")

        # Background regions
        left_pad  = M_METHOD_SPACING * 0.6
        right_pad = M_METHOD_SPACING * 0.6

        ax.axvspan(
            xpos_base[0] - left_pad, divider_x,
            facecolor=BASELINE_REGION_COLOR, alpha=BASELINE_REGION_ALPHA,
            zorder=0,
        )
        ax.axvspan(
            divider_x, xpos_wav[-1] + right_pad,
            facecolor=PROPOSED_REGION_COLOR, alpha=PROPOSED_REGION_ALPHA,
            zorder=0,
        )
        ax.axvline(
            divider_x, color=REGION_BORDER_COLOR,
            linestyle="--", linewidth=1.0, alpha=0.7, zorder=1,
        )

        # Bars: per method, one bar per dataset
        offsets = (np.arange(n_datasets) - (n_datasets - 1) / 2.0) * M_BAR_WIDTH

        for method_idx, method in enumerate(METHOD_ORDER):
            method_x = all_xpos[method_idx]
            is_wspi  = (method == "WSPI")

            for ds_idx, ds in enumerate(datasets):
                val = self._get_value(ds, method, metric)

                bar_kw: dict = dict(
                    width=M_BAR_WIDTH,
                    color=DATASET_COLORS[ds],
                    zorder=3,
                )
                if is_wspi:
                    bar_kw.update(
                        edgecolor=WSPI_EDGE_COLOR,
                        linewidth=WSPI_EDGE_LW,
                    )
                else:
                    bar_kw.update(edgecolor="white", linewidth=0.5)

                ax.bar(method_x + offsets[ds_idx], val, **bar_kw)

        # X-tick labels (proposed methods coloured + bold)
        ax.set_xticks(all_xpos)
        labels = ax.set_xticklabels(METHOD_ORDER, fontsize=10)
        for tick_label, method in zip(labels, METHOD_ORDER):
            if method in WAVELET_METHODS:
                tick_label.set_color(WAVELET_TICK_COLOR)
                tick_label.set_fontweight("bold")

        # Cosmetics
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(xpos_base[0] - left_pad, xpos_wav[-1] + right_pad)
        if metric != "delta_rank":
            ax.set_ylim(0, 1.10)

        # "Proposed Methods" label
        ymin, ymax = ax.get_ylim()
        ax.text(
            (divider_x + xpos_wav[-1] + right_pad) / 2.0,
            ymax * 0.98,
            "Proposed Methods",
            ha="center", va="top",
            color=WAVELET_TICK_COLOR, fontweight="bold", fontsize=10,
        )

        # Legend (one entry per dataset) — placed BELOW the plot, outside axes
        legend_handles = [
            mpatches.Patch(facecolor=DATASET_COLORS[ds], label=ds)
            for ds in datasets
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            fontsize=9,
            frameon=False,
            ncol=len(datasets),
            title="Dataset",
            title_fontsize=9,
        )

        self._save_fig(fig, savepath, show=show)

    # ──────────────────────────────────────────────────────────────────────────
    # Mode 2: group_by="dataset"
    # ──────────────────────────────────────────────────────────────────────────

    def _plot_by_dataset(
        self,
        metric: str,
        savepath: Union[Path, str],
        show: bool = False,
    ) -> None:
        datasets   = self._ordered_datasets()
        n_datasets = len(datasets)
        n_methods  = len(METHOD_ORDER)

        # X positions: one group per dataset
        group_centers = np.arange(n_datasets) * D_DATASET_SPACING
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * D_BAR_WIDTH

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.set_facecolor("#EAEAF2")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=GRID_ALPHA, linewidth=1.0, color="white")

        # Bars: per dataset, one bar per method
        for method_idx, method in enumerate(METHOD_ORDER):
            is_wavelet = method in WAVELET_METHODS

            xs = group_centers + offsets[method_idx]
            ys = [self._get_value(ds, method, metric) for ds in datasets]

            bar_kw: dict = dict(
                width=D_BAR_WIDTH,
                color=METHOD_COLORS[method],
                zorder=3,
            )
            if is_wavelet:
                bar_kw.update(
                    edgecolor=PROPOSED_EDGE_COL,
                    linewidth=PROPOSED_EDGE_LW,
                    hatch=PROPOSED_HATCH,
                )
            else:
                bar_kw.update(edgecolor="white", linewidth=0.4)

            ax.bar(xs, ys, **bar_kw)

        # Dataset x-tick labels
        ax.set_xticks(group_centers)
        ax.set_xticklabels(datasets, fontsize=10)

        # Cosmetics
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        x_pad = D_DATASET_SPACING * 0.55
        ax.set_xlim(group_centers[0] - x_pad, group_centers[-1] + x_pad)
        if metric != "delta_rank":
            ax.set_ylim(0, 1.10)

        # Legend: one entry per method, proposed-method labels bold + coloured
        legend_handles = []
        for method in METHOD_ORDER:
            is_wavelet = method in WAVELET_METHODS
            patch_kw: dict = dict(
                facecolor=METHOD_COLORS[method],
                label=method,
            )
            if is_wavelet:
                patch_kw.update(
                    edgecolor=PROPOSED_EDGE_COL,
                    linewidth=PROPOSED_EDGE_LW,
                    hatch=PROPOSED_HATCH,
                )
            legend_handles.append(mpatches.Patch(**patch_kw))

        legend = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            fontsize=9,
            frameon=False,
            ncol=9,
            title="Method",
            title_fontsize=9,
            handlelength=1.4,
            columnspacing=1.0,
        )
        # Style proposed method legend texts in bold + colour
        for text, method in zip(legend.get_texts(), METHOD_ORDER):
            if method in WAVELET_METHODS:
                text.set_color(WAVELET_TICK_COLOR)
                text.set_fontweight("bold")

        self._save_fig(fig, savepath, show=show)

    # ──────────────────────────────────────────────────────────────────────────
    # Save helper & public API
    # ──────────────────────────────────────────────────────────────────────────

    _shown_agg_warning = False

    @staticmethod
    def _save_fig(fig, savepath: Union[Path, str], show: bool = False) -> None:
        # bbox_inches="tight" will expand the saved bounding box to include the
        # external legend below the axes, so we don't need extra padding.
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=FIG_DPI, bbox_inches="tight")

        if show:
            backend = matplotlib.get_backend().lower()
            if backend == "agg":
                if not CrossDatasetVisualizer._shown_agg_warning:
                    import warnings
                    warnings.warn(
                        "show=True was requested but the matplotlib backend "
                        "is 'Agg' (non-interactive).  Set CROSS_VIZ_BACKEND="
                        "TkAgg (or another interactive backend) BEFORE "
                        "running this script to display figures.  "
                        "Figures are still being saved to disk normally.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    CrossDatasetVisualizer._shown_agg_warning = True
            else:
                plt.show()

        plt.close(fig)

    def _make_bar_chart(
        self,
        metric: str,
        savepath: Union[Path, str],
        group_by: str = "method",
        show: bool = False,
    ) -> None:
        """Dispatch to the appropriate plotting routine based on group_by."""
        if group_by == "method":
            self._plot_by_method(metric, savepath, show=show)
        elif group_by == "dataset":
            self._plot_by_dataset(metric, savepath, show=show)
        else:
            raise ValueError(
                f"group_by must be 'method' or 'dataset', got {group_by!r}"
            )

    # ── One-metric public methods ────────────────────────────────────────────

    def plot_ndcg10(self, savepath, group_by: str = "method",
                    show: bool = False) -> None:
        self._make_bar_chart("ndcg_10", savepath, group_by=group_by, show=show)

    def plot_spearman(self, savepath, group_by: str = "method",
                      show: bool = False) -> None:
        self._make_bar_chart("spearman_rho", savepath, group_by=group_by, show=show)

    def plot_rsi10(self, savepath, group_by: str = "method",
                   show: bool = False) -> None:
        self._make_bar_chart("rsi_10", savepath, group_by=group_by, show=show)

    def plot_deltarank(self, savepath, group_by: str = "method",
                       show: bool = False) -> None:
        self._make_bar_chart("delta_rank", savepath, group_by=group_by, show=show)

    def plot_all(
        self,
        output_dir: Union[Path, str],
        group_by: str = "method",
        show: bool = False,
    ) -> None:
        """
        Generate all four metric figures.

        Args:
            output_dir: target directory
            group_by:   "method" (default) or "dataset"
            show:       if True, also display each figure in a viewer window
                        (requires an interactive matplotlib backend; set
                        environment variable CROSS_VIZ_BACKEND=tk or similar
                        before importing this module)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.plot_ndcg10  (out / METRIC_FILENAMES["ndcg_10"],      group_by=group_by, show=show)
        self.plot_spearman(out / METRIC_FILENAMES["spearman_rho"], group_by=group_by, show=show)
        self.plot_rsi10   (out / METRIC_FILENAMES["rsi_10"],       group_by=group_by, show=show)
        self.plot_deltarank(out / METRIC_FILENAMES["delta_rank"],  group_by=group_by, show=show)

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-scenario thesis figures
    # ---------------------------------------------------------------------------
    # Consolidated from the standalone scripts viz_additions.py (ablation,
    # sensitivity, temporal-scale) and generate_granularity_trend.py
    # (granularity trend).
    #
    # ablation_components / sensitivity_curves need their OWN separate summary
    # CSVs: their method variants (WSPI-noR, WSPI-noWE, WSPI-DWT for ablation;
    # WSPI_a0.25, WSPI_b1.5, ... for sensitivity) come from dedicated
    # experiment runs and never appear in the standard 9-method protocol
    # output for the 4 main datasets, so they stay static methods with
    # explicit csv_youtube / csv_taxi parameters and their own darkgrid style.
    #
    # temporal_scale_robustness / granularity_trend, by contrast, only need
    # RSI@10 / ΔRank for the standard methods across the three NYC Yellow Taxi
    # granularities — data that's already sitting in self._agg once the
    # constructor is given the four dataset addresses. They are instance
    # methods that reuse the class's own data pipeline and style constants
    # (no separate input files at all).
    # ──────────────────────────────────────────────────────────────────────────

    # The three NYC Yellow Taxi granularities, in order, with short x-axis
    # labels — shared by temporal_scale_robustness and granularity_trend.
    _TAXI_GRANULARITY_DATASETS: list[tuple[str, str]] = [
        ("NYC Yellow Taxi Hourly", "Hourly"),
        ("NYC Yellow Taxi 30m", "30-min"),
        ("NYC Yellow Taxi 5m", "5-min"),
    ]

    def _require_taxi_granularities(self) -> list[tuple[str, str]]:
        missing = [ds for ds, _ in self._TAXI_GRANULARITY_DATASETS
                   if ds not in self._raw]
        if missing:
            raise ValueError(
                "This figure requires the three NYC Yellow Taxi granularities "
                f"(Hourly, 30m, 5m) to be loaded; missing: {missing}. Pass "
                "them via the dataset_results dict given to the constructor."
            )
        return self._TAXI_GRANULARITY_DATASETS

    @staticmethod
    def plot_ablation_components(
        csv_youtube: Union[Path, str] = "results/ablation/ablation_summary_youtube.csv",
        csv_taxi: Union[Path, str] = "results/ablation/ablation_summary_yellowtaxi_1h.csv",
        savepath: Union[Path, str] = "results/ablation/chart19_ablation_components.png",
        show: bool = False,
    ) -> str:
        """Chart 19 — grouped ablation bars (RSI@10 and ΔRank) for the four WSPI
        variants on YouTube and NYC Yellow Taxi (hourly).  Standalone
        counterpart: ``viz_additions.chart19_ablation_components``."""
        with plt.style.context(_SCENARIO_STYLE):
            dfs = {
                "YouTube Hourly": pd.read_csv(csv_youtube).set_index("method"),
                "NYC Yellow Taxi Hourly": pd.read_csv(csv_taxi).set_index("method"),
            }
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
            panels = [
                ("rsi@10", "RSI@10  (higher = better ↑)", axes[0]),
                ("robustness_distortion", "ΔRank  (lower = better ↓)", axes[1]),
            ]
            x, width = np.arange(len(dfs)), 0.19
            for mcol, ylabel, ax in panels:
                for i, variant in enumerate(_ABL_ORDER):
                    vals = [dfs[d].loc[variant, mcol] for d in dfs]
                    bars = ax.bar(
                        x + (i - 1.5) * width, vals, width,
                        label=_ABL_LABELS[variant], color=_ABL_COLORS[variant],
                        edgecolor="black", linewidth=0.5,
                        hatch="//" if variant == "WSPI" else None,
                    )
                    for b, v in zip(bars, vals):
                        ax.annotate(
                            f"{v:.3g}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8,
                        )
                ax.set_xticks(x)
                ax.set_xticklabels(list(dfs.keys()), fontsize=10)
                ax.set_ylabel(ylabel)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                loc="upper center", bbox_to_anchor=(0.5, 0.02),
                ncol=4, frameon=False, fontsize=8.5,
                handlelength=1.6, columnspacing=1.4,
            )
            fig.suptitle(
                "Ablation of WSPI Components — contribution of R, WE, and DTCWT",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0.08, 1, 0.97))
        CrossDatasetVisualizer._save_fig(fig, savepath, show=show)
        return str(savepath)

    @staticmethod
    def plot_sensitivity_curves(
        csv_youtube: Union[Path, str] = "results/sensitivity/sensitivity_summary_youtube.csv",
        csv_taxi: Union[Path, str] = "results/sensitivity/sensitivity_summary_yellowtaxi_1h.csv",
        savepath: Union[Path, str] = "results/sensitivity/chart20_sensitivity_curves.png",
        show: bool = False,
    ) -> str:
        """Chart 20 — 2x2 sensitivity grid: RSI@10 and ΔRank versus α and β
        (one-dimensional sweep, other coefficient fixed at 1), both datasets.
        Standalone counterpart: ``viz_additions.chart20_sensitivity_curves``."""

        def load(csv):
            rows = []
            for _, r in pd.read_csv(csv).iterrows():
                tag = r["method"].split("_")[1]          # a0.25 / b1.5
                rows.append(dict(coef=tag[0], val=float(tag[1:]),
                                 rsi=r["rsi@10"], dr=r["robustness_distortion"]))
            return pd.DataFrame(rows)

        with plt.style.context(_SCENARIO_STYLE):
            data = {"YouTube Hourly": load(csv_youtube),
                    "NYC Yellow Taxi Hourly": load(csv_taxi)}
            ds_style = {
                "YouTube Hourly": dict(color=_SCENARIO_METHOD_COLORS["WSPI"], marker="o"),
                "NYC Yellow Taxi Hourly": dict(color=_SCENARIO_METHOD_COLORS["AF"], marker="s"),
            }
            fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
            for col, coef in enumerate(["a", "b"]):
                sym = "α" if coef == "a" else "β"
                for row, (mkey, ylabel) in enumerate(
                        [("rsi", "RSI@10 ↑"), ("dr", "ΔRank ↓")]):
                    ax = axes[row, col]
                    for ds, d in data.items():
                        dd = d[d.coef == coef].sort_values("val")
                        ax.plot(dd.val, dd[mkey], lw=2, label=ds, **ds_style[ds])
                    ax.axvline(1.0, color="gray", ls="--", lw=1)
                    if row == 1:
                        ax.set_xlabel(f"{sym}  (other coefficient fixed at 1)")
                    if col == 0:
                        ax.set_ylabel(ylabel)
            axes[0, 0].legend(fontsize=8, loc="center right")
            fig.suptitle(
                "Sensitivity of WSPI to fusion coefficients "
                "— flat RSI, ΔRank minimised near 1", fontsize=12,
            )
            fig.tight_layout()
        CrossDatasetVisualizer._save_fig(fig, savepath, show=show)
        return str(savepath)

    def plot_temporal_scale_robustness(
        self,
        savepath: Union[Path, str] = "results/chart16_temporal_scale_robustness.png",
        show: bool = False,
    ) -> str:
        """Chart 16 (data-driven) — NYC Yellow Taxi temporal-scale robustness.
        Plots the corrected, monotonic RSI@10 / ΔRank trends across the three
        taxi granularities (the version embedded in the old ``visualizer.py``
        hard-coded stale pre-bugfix numbers). Sourced entirely from
        ``self._agg`` — i.e. the same four dataset addresses passed to the
        constructor, no separate CSV files. Standalone counterpart:
        ``viz_additions.chart16_temporal_scale_robustness_FIXED``."""
        granularities = self._require_taxi_granularities()
        methods = ["AF", "EWMA", "RRD", "DWT+AF", "DTCWT+AF", "WSPI"]
        gx = list(range(len(granularities)))

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        for ax, metric, ylabel in [
                (axes[0], "rsi_10", "RSI@10  (higher = better ↑)"),
                (axes[1], "delta_rank", "ΔRank  (lower = better ↓)")]:
            for m in methods:
                vals = [self._get_value(ds, m, metric) for ds, _ in granularities]
                ax.plot(gx, vals, marker="o", lw=2.4 if m == "WSPI" else 1.4,
                        color=METHOD_COLORS[m], label=m,
                        zorder=5 if m == "WSPI" else 2)
                if m == "WSPI":
                    for xx, yy in zip(gx, vals):
                        ax.annotate(
                            f"{yy:.3g}", (xx, yy), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=8,
                            color=METHOD_COLORS["WSPI"], fontweight="bold")
            ax.set_xticks(gx)
            ax.set_xticklabels([short for _, short in granularities])
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(
            "NYC Yellow Taxi — temporal-scale robustness "
            "(data-driven, post-bugfix values)", fontsize=12,
        )
        fig.tight_layout()
        self._save_fig(fig, savepath, show=show)
        return str(savepath)

    # ── Granularity-trend line chart (Figure 8) ───────────────────────────────

    _GRANULARITY_TREND_TRADITIONAL_EXCLUDE = "PFRF"  # degenerate baseline

    def _best_traditional(self, dataset: str, metric: str) -> float:
        traditional = [m for m in BASELINE_METHODS
                       if m != self._GRANULARITY_TREND_TRADITIONAL_EXCLUDE]
        vals = [self._get_value(dataset, m, metric) for m in traditional]
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            return float("nan")
        return max(vals) if metric == "rsi_10" else min(vals)

    def plot_granularity_trend(
        self,
        savepath: Union[Path, str] = "results/fig8_granularity_trend.png",
        show: bool = False,
    ) -> str:
        """Figure 8 — WSPI performance versus temporal granularity on NYC
        Yellow Taxi (Hourly -> 30-min -> 5-min): a two-panel line-trend chart
        of (a) RSI@10 rising and (b) ΔRank falling as granularity gets finer,
        for WSPI, DTCWT+AF, and a "Best traditional" reference (best baseline
        per granularity, excluding the degenerate PFRF). Sourced entirely
        from ``self._agg`` — i.e. the same four dataset addresses passed to
        the constructor, no separate CSV files. Standalone counterpart:
        ``scripts/generate_granularity_trend.py``."""
        granularities = self._require_taxi_granularities()
        x = np.arange(len(granularities))
        xticklabels = [short for _, short in granularities]

        line_style = {
            "WSPI": dict(color=METHOD_COLORS["WSPI"], ls="-", lw=2.6, marker="o",
                        ms=7.5, markeredgecolor=PROPOSED_EDGE_COL,
                        markeredgewidth=0.9, zorder=6),
            "DTCWT+AF": dict(color=METHOD_COLORS["DTCWT+AF"], ls="--", lw=2.0,
                             marker="s", ms=6.5, markeredgecolor=PROPOSED_EDGE_COL,
                             markeredgewidth=0.7, zorder=5),
            "Best traditional": dict(color="#888888", ls=":", lw=1.9, marker="^",
                                     ms=6.5, zorder=4),
        }

        def series(metric):
            wspi, dtcwt, best = [], [], []
            for ds, _short in granularities:
                wspi.append(self._get_value(ds, "WSPI", metric))
                dtcwt.append(self._get_value(ds, "DTCWT+AF", metric))
                best.append(self._best_traditional(ds, metric))
            return wspi, dtcwt, best

        def draw_panel(ax, series_map, ylabel, title, annotate_fmt, annotate_below):
            ax.set_facecolor("#EAEAF2")
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, alpha=GRID_ALPHA, linewidth=1.0, color="white")
            for name in ("WSPI", "DTCWT+AF", "Best traditional"):
                ax.plot(x, series_map[name], label=name, **line_style[name])
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

        rsi_w, rsi_d, rsi_b = series("rsi_10")
        dr_w, dr_d, dr_b = series("delta_rank")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

        draw_panel(
            ax1, {"WSPI": rsi_w, "DTCWT+AF": rsi_d, "Best traditional": rsi_b},
            METRIC_LABELS["rsi_10"], "(a) Temporal stability",
            annotate_fmt="{:.3f}", annotate_below=False,
        )
        ax1.set_ylim(0, max(rsi_w + rsi_d) * 1.18)

        draw_panel(
            ax2, {"WSPI": dr_w, "DTCWT+AF": dr_d, "Best traditional": dr_b},
            METRIC_LABELS["delta_rank"], "(b) Noise robustness",
            annotate_fmt="{:.2f}", annotate_below=True,
        )
        ax2.set_ylim(0, max(dr_d + dr_b) * 1.15)

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

        self._save_fig(fig, savepath, show=show)
        return str(savepath)

    # ── Convenience: generate every cross-scenario figure at once ─────────────

    _SCENARIO_FIGURE_FILENAMES: dict[str, str] = {
        "ablation_components": "chart19_ablation_components.png",
        "sensitivity_curves": "chart20_sensitivity_curves.png",
        "temporal_scale_robustness": "chart16_temporal_scale_robustness.png",
        "granularity_trend": "fig8_granularity_trend.png",
    }

    def plot_all_scenario_figures(
        self,
        out_dir: Union[Path, str],
        ablation_csv_youtube: Union[Path, str] = "results/ablation/ablation_summary_youtube.csv",
        ablation_csv_taxi: Union[Path, str] = "results/ablation/ablation_summary_yellowtaxi_1h.csv",
        sensitivity_csv_youtube: Union[Path, str] = "results/sensitivity/sensitivity_summary_youtube.csv",
        sensitivity_csv_taxi: Union[Path, str] = "results/sensitivity/sensitivity_summary_yellowtaxi_1h.csv",
        show: bool = False,
    ) -> dict:
        """Generate all four cross-scenario thesis figures, all saved into a
        single ``out_dir`` (no per-figure output path).

        ``temporal_scale_robustness`` and ``granularity_trend`` need no extra
        arguments — they're derived from the same four dataset addresses
        already given to the constructor. ``ablation_components`` and
        ``sensitivity_curves`` need their own separate summary CSVs (distinct
        experiment runs with method variants that aren't part of the main
        protocol output); pass them explicitly if they don't live at the
        default paths above.

        Missing input files are skipped with a message rather than failing
        the whole run. Returns a mapping of figure name -> saved path (or
        None if skipped)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filenames = CrossDatasetVisualizer._SCENARIO_FIGURE_FILENAMES

        jobs = [
            ("ablation_components", lambda sp: CrossDatasetVisualizer.plot_ablation_components(
                csv_youtube=ablation_csv_youtube, csv_taxi=ablation_csv_taxi,
                savepath=sp, show=show)),
            ("sensitivity_curves", lambda sp: CrossDatasetVisualizer.plot_sensitivity_curves(
                csv_youtube=sensitivity_csv_youtube, csv_taxi=sensitivity_csv_taxi,
                savepath=sp, show=show)),
            ("temporal_scale_robustness",
             lambda sp: self.plot_temporal_scale_robustness(savepath=sp, show=show)),
            ("granularity_trend",
             lambda sp: self.plot_granularity_trend(savepath=sp, show=show)),
        ]
        results: dict = {}
        for name, fn in jobs:
            savepath = out_dir / filenames[name]
            try:
                results[name] = fn(savepath)
                print("saved:", results[name])
            except (FileNotFoundError, ValueError) as e:
                results[name] = None
                print("skipped", name, "-", e)
        return results

    # ── Summary table ─────────────────────────────────────────────────────────

    def save_summary_table(self, savepath: Union[Path, str]) -> None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)

        ds_rank = {d: i for i, d in enumerate(self._ordered_datasets())}
        m_rank  = {m: i for i, m in enumerate(METHOD_ORDER)}

        out = self._agg.copy()
        out["_ds"] = out["dataset"].map(ds_rank)
        out["_m"]  = out["method"].map(m_rank)
        out = out.sort_values(["_ds", "_m"]).drop(columns=["_ds", "_m"])
        out.to_csv(savepath, index=False)
