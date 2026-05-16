"""
Cross-dataset comparison visualizer for the WSPI paper.

Aggregates per-window results from multiple completed dataset runs and
produces one grouped-bar figure per metric so that each figure shows ONE
metric across ALL datasets (Figures 3, 5, 6, 7 in the paper).

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Method ordering (left to right within every group) ───────────────────────

METHOD_ORDER: list[str] = [
    "AF", "EWMA", "RRD", "VSE", "CompoundPop", "PFRF",   # baselines
    "DWT+AF", "DTCWT+AF", "WSPI",                         # wavelet
]

WAVELET_METHODS: set[str] = {"DWT+AF", "DTCWT+AF", "WSPI"}

# ── Colour / style constants (tweak here without touching any other code) ─────

BASELINE_COLOR = "#B0C4DE"
BASELINE_ALPHA = 0.55

WAVELET_COLORS: dict[str, str] = {
    "DWT+AF":   "#FF8C00",
    "DTCWT+AF": "#4682B4",
    "WSPI":     "#C71585",
}
WAVELET_ALPHA = 1.0

WSPI_HATCH   = "//"      # hatching applied to WSPI bars so they pop as "ours"
WSPI_EDGE_LW = 1.8       # black edge linewidth on WSPI bars

# ── Dataset display order ─────────────────────────────────────────────────────

DATASET_ORDER: list[str] = [
    "YouTube Hourly",
    "Uber Hourly",
    "Uber 30m",
    "Uber 5m",
]

# ── Column name aliases ───────────────────────────────────────────────────────
# Maps every variant that temporal_evaluator.py might write → canonical name.

_COL_ALIASES: dict[str, str] = {
    "ndcg@10":               "ndcg_10",
    "ndcg_10":               "ndcg_10",
    "rsi@10":                "rsi_10",
    "rsi_10":                "rsi_10",
    "robustness_distortion": "delta_rank",   # actual saved name
    "delta_rank":            "delta_rank",
    "spearman_rho":          "spearman_rho",
}

_CANONICAL_COLS: list[str] = ["ndcg_10", "spearman_rho", "rsi_10", "delta_rank"]

# ── Axis label text ───────────────────────────────────────────────────────────

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

# ── Figure layout constants ───────────────────────────────────────────────────

BAR_WIDTH  = 0.08   # data-unit width of each bar
GROUP_GAP  = 0.28   # blank space between dataset groups
FIG_SIZE   = (11, 5)
FIG_DPI    = 300
GRID_ALPHA = 0.25


# ─────────────────────────────────────────────────────────────────────────────

class CrossDatasetVisualizer:
    """Aggregate per-window results from multiple datasets and plot per-metric."""

    def __init__(
        self,
        dataset_results: dict,          # {label: DataFrame | Path | str}
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
            # temporal_evaluator.py stores one file per method under protocol/
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
        """Return a DataFrame indexed by (dataset, method) with mean metric values."""
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
                        # Exclude LRU from spearman aggregation (broken rank corr.)
                        # LRU is not in METHOD_ORDER but guard here for robustness.
                        if col == "spearman_rho":
                            series = mdf.loc[mdf["method"] != "LRU", col].dropna()
                        else:
                            series = mdf[col].dropna()
                        row[col] = float(series.mean()) if len(series) else np.nan

                rows.append(row)

        return pd.DataFrame(rows)

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _ordered_datasets(self) -> list[str]:
        present = list(self._raw.keys())
        ordered = [d for d in DATASET_ORDER if d in present]
        ordered += [d for d in present if d not in ordered]
        return ordered

    def _make_bar_chart(self, metric: str, savepath: Union[Path, str]) -> None:
        datasets  = self._ordered_datasets()
        n_groups  = len(datasets)
        n_methods = len(METHOD_ORDER)

        group_width   = n_methods * BAR_WIDTH + GROUP_GAP
        group_centers = np.arange(n_groups) * group_width

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, alpha=GRID_ALPHA, linewidth=0.8, color="grey")

        legend_handles: list[mpatches.Patch] = []

        for i, method in enumerate(METHOD_ORDER):
            offset = (i - (n_methods - 1) / 2) * BAR_WIDTH
            xpos   = group_centers + offset

            values: list[float] = []
            for ds in datasets:
                mask = (self._agg["dataset"] == ds) & (self._agg["method"] == method)
                val  = self._agg.loc[mask, metric]
                values.append(
                    float(val.iloc[0]) if not val.empty and not val.isna().all()
                    else np.nan
                )

            is_wavelet = method in WAVELET_METHODS
            is_wspi    = method == "WSPI"

            bar_kw: dict = dict(width=BAR_WIDTH, zorder=2)
            if is_wavelet:
                bar_kw.update(color=WAVELET_COLORS[method], alpha=WAVELET_ALPHA)
            else:
                bar_kw.update(color=BASELINE_COLOR, alpha=BASELINE_ALPHA)

            if is_wspi:
                bar_kw.update(
                    hatch=WSPI_HATCH,
                    edgecolor="black",
                    linewidth=WSPI_EDGE_LW,
                )

            bars = ax.bar(xpos, values, **bar_kw)

            # Build matching legend patch
            patch_kw: dict = dict(
                label=method,
                facecolor=bar_kw["color"],
                alpha=bar_kw.get("alpha", 1.0),
            )
            if is_wspi:
                patch_kw.update(
                    hatch=WSPI_HATCH,
                    edgecolor="black",
                    linewidth=WSPI_EDGE_LW,
                )
            legend_handles.append(mpatches.Patch(**patch_kw))

            # Annotate WSPI bars with value labels (rotated to fit narrow bars)
            if is_wspi:
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}",
                            ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold",
                            rotation=90,
                        )

        ax.set_xticks(group_centers)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            ncol=1,
        )

        fig.tight_layout()

        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    # ── Public plotting API ───────────────────────────────────────────────────

    def plot_ndcg10(self, savepath: Union[Path, str]) -> None:
        self._make_bar_chart("ndcg_10", savepath)

    def plot_spearman(self, savepath: Union[Path, str]) -> None:
        self._make_bar_chart("spearman_rho", savepath)

    def plot_rsi10(self, savepath: Union[Path, str]) -> None:
        self._make_bar_chart("rsi_10", savepath)

    def plot_deltarank(self, savepath: Union[Path, str]) -> None:
        self._make_bar_chart("delta_rank", savepath)

    def plot_all(self, output_dir: Union[Path, str]) -> None:
        """Generate all four metric figures into ``output_dir``."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.plot_ndcg10(out / METRIC_FILENAMES["ndcg_10"])
        self.plot_spearman(out / METRIC_FILENAMES["spearman_rho"])
        self.plot_rsi10(out / METRIC_FILENAMES["rsi_10"])
        self.plot_deltarank(out / METRIC_FILENAMES["delta_rank"])

    # ── Summary table ─────────────────────────────────────────────────────────

    def save_summary_table(self, savepath: Union[Path, str]) -> None:
        """Write aggregated values (4 datasets × 9 methods × 4 metrics) to CSV."""
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)

        ds_rank = {d: i for i, d in enumerate(self._ordered_datasets())}
        m_rank  = {m: i for i, m in enumerate(METHOD_ORDER)}

        out = self._agg.copy()
        out["_ds"] = out["dataset"].map(ds_rank)
        out["_m"]  = out["method"].map(m_rank)
        out = out.sort_values(["_ds", "_m"]).drop(columns=["_ds", "_m"])
        out.to_csv(savepath, index=False)
