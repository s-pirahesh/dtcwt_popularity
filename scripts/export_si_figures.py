r"""
Export the figures of the Supplementary Information of paper V5 at their printed size
====================================================================================
Same method as scripts/export_paper_figures.py (chat 27): the figure functions of
scripts/generate_revision_figures.py are called unchanged, inside the
``paper_mode`` wrapper of export_paper_figures.py, so that every figure is drawn
at the width it has in WSPI_SI.tex, with text of 7-9 pt embedded as TrueType
(Type 42).  No number is computed here.

Added for paper V5 (chat 28, 28 Sep 2026; decision in chat 27: "the SI figures
are also made at printed size").  No existing file of the project is changed.

  SI figure  function                        source (results/revision_v5)
  figS1      fig_t22_window_curves           T2.2_window_sweep/sweep_summary.csv
  figS2      fig_t37_feature_relation        T3.7_feature_relation/<s>/density.csv, pooled_summary.csv
  figS3      fig_t35_spike_size              T3.5_robustness/robustness_summary.csv
  figS4      fig_t38_runtime                 T3.8_runtime/bench/runtime_grid.csv, memory/tracemalloc_grid.csv
  figS5      fig_t36_shift_invariance        T3.6_shift_invariance/shift_summary.csv, synthetic/synthetic_summary.csv
  figS6      fig_t32_param_heatmap           T3.2_param_grid/grid_summary.csv, selection/selection.csv
  figS7      fig_t310_ml_rsi_time            T3.9_movielens/default/<d>/protocol/
  figS8      fig_t24_delay_ecdf              T2.4_responsiveness/<s>/<cfg>/delays.csv

Usage
-----
  set PYTHONDONTWRITEBYTECODE=1
  python scripts\export_si_figures.py --results results\revision_v5 --out <folder>
  optional: --only figS1 figS8 ...
Output: <out>/figS1.pdf ... figS8.pdf, the same as .png (600 dpi) and
<out>/export_si_figures_run.json.  Nothing is written to results.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_paper_figures as epf  # noqa: E402
import generate_revision_figures as g  # noqa: E402

LANDSCAPE_WIDTH_IN = 9.0      # text height of the letter page, used for the landscape SI figures

# SI name: (function, printed width in, height in, legend columns, landscape page)
SI_FIGURES = {
    'figS1': (g.fig_t22_window_curves, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
    'figS2': (g.fig_t37_feature_relation, 6.5, 2.4, 2, False),
    'figS3': (g.fig_t35_spike_size, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
    'figS4': (g.fig_t38_runtime, 6.5, 2.7, 5, False),
    'figS5': (g.fig_t36_shift_invariance, LANDSCAPE_WIDTH_IN, 5.2, 3, True),
    'figS6': (g.fig_t32_param_heatmap, LANDSCAPE_WIDTH_IN, 6.2, 2, True),
    'figS7': (g.fig_t310_ml_rsi_time, 6.5, 4.6, 6, False),
    'figS8': (g.fig_t24_delay_ecdf, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
}

# long panel titles that the caption repeats: shortened to the panel letter
SHORT_TITLES = {'figS4', 'figS5'}
# per-panel text boxes that Supplementary Table S2 repeats
DROP_AXES_TEXT = {'figS2'}
# cell labels of the heat maps: kept below the 7 pt floor, so that they fit the cells
CELL_LABEL_PT = 5.0


def si_fix(name, fig, ncols):
    """Layout of one SI figure at printed size; no data is touched.
    - the suptitle is removed (the caption carries it);
    - legends drawn below the panels become one legend outside the panels
      (constrained layout then keeps it clear of the axis labels);
    - listed figures: long panel titles shortened to '(a)', text boxes removed."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    handles, labels = [], []
    for leg in list(fig.legends):
        for h, t in zip(leg.legend_handles, leg.get_texts()):
            if t.get_text() not in labels:
                handles.append(h)
                labels.append(t.get_text())
        leg.remove()
    if name in ('figS2', 'figS4'):
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            for h, t in zip(leg.legend_handles, leg.get_texts()):
                lab = t.get_text()
                # the same method may be labelled 'WSPI (N=64)' in one panel and 'WSPI' in another
                if lab.split(' (')[0] not in [x.split(' (')[0] for x in labels]:
                    handles.append(h)
                    labels.append(lab)
            leg.remove()
    if handles:
        fig.legend(handles, labels, loc='outside lower center', ncols=ncols, frameon=False,
                   fontsize=epf.FONT_MIN + 0.5)
    for ax in fig.axes:
        yl, xl = ax.get_ylabel(), ax.get_xlabel()
        if name in ('figS3', 'figS8') and yl:
            first = 'Default' if 'default' in yl.lower() else 'Equal window of 64' if 'equal' in yl.lower() else ''
            second = '$\\Delta$Rank (log scale)' if name == 'figS3' else 'share of entries in Top-10'
            ax.set_ylabel(f'{first}\n{second}' if first else second)
        if name == 'figS3' and xl:
            ax.set_xlabel('spike size (x mean of 64 slots)')
        if name == 'figS2':
            if xl:
                ax.set_xlabel('$R$')
            if yl.startswith('WE'):
                ax.set_ylabel('$W_E$')
        if name == 'figS4' and yl:
            ax.set_ylabel({'(a)': 'seconds per window', '(b)': 'microseconds per item-window',
                           '(c)': 'traced bytes per item'}.get(ax.get_title()[:3], yl))
        if name == 'figS5' and len(ax.get_xticklabels()) and any(
                t.get_text().startswith('level') for t in ax.get_xticklabels()):
            ax.set_xticks(ax.get_xticks(), [t.get_text().replace('lowpass', 'low').replace('level ', '')
                                            for t in ax.get_xticklabels()])
            ax.set_xlabel('band')
        if name == 'figS6':
            for axis in (ax.xaxis, ax.yaxis):
                labs = [t.get_text() for t in axis.get_ticklabels()]
                if labs:
                    axis.set_ticks(axis.get_ticklocs(), [l.replace('0.', '.') if l.startswith('0.') else l
                                                          for l in labs])
        if name in SHORT_TITLES:
            t = ax.get_title()
            if t.startswith('(') and ')' in t:
                ax.set_title(t[:t.index(')') + 1])
        if name in DROP_AXES_TEXT:
            for tx in list(ax.texts):
                tx.remove()
        if name == 'figS6':
            for tx in ax.texts:     # bypasses the 7-9 pt clip of paper_mode on purpose
                tx.get_fontproperties()._size = CELL_LABEL_PT
                if tx.get_text().startswith('0.'):    # '0.9667' -> '.9667', so that it fits the cell
                    tx.set_text(tx.get_text()[1:])


class si_tidy:
    """Run si_fix after the tidy of export_paper_figures, inside its save step."""

    def __init__(self, name, ncols):
        self.name, self.ncols = name, ncols

    def __enter__(self):
        self.orig = epf.tidy

        def tidy(fig):
            self.orig(fig)
            si_fix(self.name, fig, self.ncols)
        epf.tidy = tidy

    def __exit__(self, *exc):
        epf.tidy = self.orig


class no_tight_layout:
    """The figures are drawn with the constrained layout of paper_mode; a call of
    Figure.tight_layout inside a figure function (which cannot switch engines once a
    colorbar exists) is skipped while an SI figure is drawn."""

    def __enter__(self):
        self.orig = Figure.tight_layout
        Figure.tight_layout = lambda self, *a, **kw: None

    def __exit__(self, *exc):
        Figure.tight_layout = self.orig


def main():
    ap = argparse.ArgumentParser(description='SI figures of paper V5 at printed size')
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = epf.FONT_MIN + 1

    meta = dict(program='scripts/export_si_figures.py',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                matplotlib=matplotlib.__version__, font_range_pt=[epf.FONT_MIN, epf.FONT_MAX],
                text_width_in=epf.TEXT_WIDTH_IN, png_dpi=epf.PNG_DPI, fonttype=42, figures={})
    ok = 0
    names = a.only or list(SI_FIGURES)
    for name in names:
        func, width, height, ncols, landscape = SI_FIGURES[name]
        size = (width, height)
        sink = {}
        with epf.paper_mode(size, ncols, sink, name), no_tight_layout(), si_tidy(name, ncols):
            res = func(a.results, a.out)
        files, note = res if isinstance(res, tuple) else (res, '')
        if not files:
            print(f'[skip] {name}: {note}')
            continue
        ok += 1
        meta['figures'][name] = dict(function=func.__name__, figsize_in=size, legend_ncols=ncols,
                                     landscape=landscape,
                                     note=note, md5={f: epf.md5(a.out / f) for f in files})
        print(f'[ok]   {name}: {", ".join(files)}')
    (a.out / 'export_si_figures_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'{ok} of {len(names)} figures written to {a.out}')


if __name__ == '__main__':
    main()
