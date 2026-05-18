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
    "Uber Hourly",
    "Uber 30m",
    "Uber 5m",
]

DATASET_COLORS: dict[str, str] = {
    "YouTube Hourly": "#1f77b4",   # blue
    "Uber Hourly":    "#2ca02c",   # green
    "Uber 30m":       "#ff7f0e",   # orange
    "Uber 5m":        "#9467bd",   # purple
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
