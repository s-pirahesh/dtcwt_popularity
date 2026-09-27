r"""
Export the main-text figures of paper V5 at their printed size
===============================================================
The figures of ``scripts/generate_revision_figures.py`` are drawn large
(12-17 inches wide) and LaTeX shrinks them to the text width, so their text
is printed at 4-7 pt.  This program draws the same figures, from the same
functions, data and style, directly at the width they have in the paper, so
that no scaling is needed and the text keeps its size:

  * every figure is created at its printed size (width in the table below =
    the \includegraphics width of WSPI_ScientificReports.tex x 6.5 in);
  * every font size is limited to the range [FONT_MIN, FONT_MAX] pt
    (the clip is idempotent, so it never compounds);
  * fonts are embedded as TrueType (Type 42), not Type 3;
  * PDF (vector) and PNG (600 dpi) are written with the paper file names.

Nothing in generate_revision_figures.py is changed: its functions are called
unchanged; only plt.subplots / plt.figure (size), FontProperties.set_size
(font clip), Legend (number of legend columns) and the save functions
(file names, formats) are wrapped while one figure is drawn.  No number is
computed here; every value comes from the loaders of that program, which
check the statistics against summary_common_windows.csv.

Added for paper V5 (chat 27, 28 Sep 2026; decision of Sajjad: "ساخت دوباره
در اندازه چاپ + Type 42").

Usage
-----
  python scripts/export_paper_figures.py --results results/revision_v5 --out <folder>
  optional: --only fig2 fig8 ...
Output: <out>/<paper name>.pdf and .png (fig2 ... fig8, fig_movielens,
fig_surge) and <out>/export_paper_figures_run.json (sizes, font range,
matplotlib version, md5 of every file).  Nothing is written to results.
Figure 1 is TikZ (V5/source/fig/fig1_pipeline.tex) and is not built here.
"""
import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, legend as mlegend
from matplotlib.text import Annotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_revision_figures as g  # noqa: E402

TEXT_WIDTH_IN = 6.5          # letter paper, 1 in margins (WSPI_ScientificReports.tex)
FONT_MIN, FONT_MAX = 7.0, 9.0
PNG_DPI = 600

# paper name: (function, printed width as share of \textwidth, height in, legend columns)
PAPER_FIGURES = {
    'fig2': (g.fig_t310_fig2, 0.92, 2.9, 5),
    'fig3': (g.fig_t310_fig3, 0.92, 2.9, 5),
    'fig4': (g.fig_t310_fig4, 0.92, 2.9, 5),
    'fig5': (g.fig_t310_fig5, 0.92, 2.9, 5),
    'fig6': (g.fig_t310_fig6, 0.92, 2.5, 3),
    'fig7': (g.fig_t310_fig7, 0.92, 2.5, 3),
    'fig8': (g.fig_t310_fig8, 1.00, 2.7, 3),
    'fig_movielens': (g.fig_t310_movielens, 0.85, 4.1, 5),
    'fig_surge': (g.fig_t24_surge_examples_2col, 0.95, 7.6, 4),
}


def _clip(size):
    return min(max(size, FONT_MIN), FONT_MAX)


def tidy(fig):
    """Label layout for the printed size; no data is touched.
    - y labels 'Metric  (note)' are broken into two lines;
    - scenario tick labels are broken into two lines ('NYC Yellow Taxi' / 'Hourly');
    - date axes: ticks every Monday ('07 May') for a period under 90 days,
      otherwise at most 7 ticks with a concise date format;
    - the legend under two-line tick labels moves down to clear them;
    - point annotations use FONT_MIN."""
    for ax in fig.axes:
        yl = ax.get_ylabel()
        if '  (' in yl:
            ax.set_ylabel(yl.replace('  (', '\n(', 1))
        labs = [t.get_text() for t in ax.get_xticklabels()]
        if any(l.startswith(('NYC Yellow Taxi ', 'YouTube ')) for l in labs):
            new = [l.replace('NYC Yellow Taxi ', 'NYC Yellow Taxi\n')
                    .replace('YouTube ', 'YouTube\n') for l in labs]
            ax.set_xticks(ax.get_xticks(), new)
            leg = ax.get_legend()
            if leg is not None:        # legend below two-line tick labels
                leg.set_bbox_to_anchor((0.5, -0.27), transform=ax.transAxes)
        if isinstance(ax.xaxis.get_major_formatter(), mdates.AutoDateFormatter):
            lo, hi = ax.get_xlim()
            if hi - lo < 90:           # short period (YouTube): ticks every Monday, '07 May'
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            else:
                loc = mdates.AutoDateLocator(maxticks=7)
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        for ch in ax.get_children():
            if isinstance(ch, Annotation):
                ch.set_fontsize(FONT_MIN)


@contextmanager
def paper_mode(size, ncols, sink, paper_name):
    """Wrap figure size, font sizes, legend columns and saving for one figure."""
    orig_subplots, orig_figure = plt.subplots, plt.figure
    orig_set_size = font_manager.FontProperties.set_size
    orig_legend_init = mlegend.Legend.__init__
    orig_save, orig_save_v4 = g.save, g._save_v4

    def subplots(*a, **kw):
        kw['figsize'] = size
        kw.setdefault('layout', 'constrained')
        return orig_subplots(*a, **kw)

    def figure(*a, **kw):
        kw['figsize'] = size
        kw.setdefault('layout', 'constrained')
        return orig_figure(*a, **kw)

    def set_size(self, s):
        orig_set_size(self, s)
        orig_set_size(self, _clip(self.get_size_in_points()))

    def legend_init(self, *a, **kw):
        if 'ncol' in kw or 'ncols' in kw:
            kw.pop('ncol', None)
            kw['ncols'] = ncols
        orig_legend_init(self, *a, **kw)

    def save_any(fig, out, name):
        tidy(fig)
        if not fig.get_layout_engine() or 'Constrained' not in type(fig.get_layout_engine()).__name__:
            fig.set_layout_engine('constrained')
        files = []
        for ext in ('pdf', 'png'):
            f = Path(out) / f'{paper_name}.{ext}'
            fig.savefig(f, dpi=PNG_DPI, bbox_inches='tight', pad_inches=0.02,
                        metadata={'CreationDate': None} if ext == 'pdf' else None)
            files.append(f.name)
        sink['source_function_name'] = name
        plt.close(fig)
        return files

    plt.subplots, plt.figure = subplots, figure
    font_manager.FontProperties.set_size = set_size
    mlegend.Legend.__init__ = legend_init
    g.save, g._save_v4 = save_any, save_any
    try:
        yield
    finally:
        plt.subplots, plt.figure = orig_subplots, orig_figure
        font_manager.FontProperties.set_size = orig_set_size
        mlegend.Legend.__init__ = orig_legend_init
        g.save, g._save_v4 = orig_save, orig_save_v4


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = FONT_MIN + 1

    meta = dict(program='scripts/export_paper_figures.py',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                matplotlib=matplotlib.__version__, font_range_pt=[FONT_MIN, FONT_MAX],
                text_width_in=TEXT_WIDTH_IN, png_dpi=PNG_DPI, fonttype=42, figures={})
    ok = 0
    names = a.only or list(PAPER_FIGURES)
    for name in names:
        func, share, height, ncols = PAPER_FIGURES[name]
        size = (round(share * TEXT_WIDTH_IN, 3), height)
        sink = {}
        with paper_mode(size, ncols, sink, name):
            files, note = func(a.results, a.out)
        if not files:
            print(f'[skip] {name}: {note}')
            continue
        ok += 1
        meta['figures'][name] = dict(function=func.__name__, figsize_in=size,
                                     legend_ncols=ncols, note=note,
                                     md5={f: md5(a.out / f) for f in files})
        print(f'[ok]   {name}: {", ".join(files)}')
    (a.out / 'export_paper_figures_run.json').write_text(json.dumps(meta, indent=2),
                                                        encoding='utf-8')
    print(f'{ok} of {len(names)} figures written to {a.out}')


if __name__ == '__main__':
    main()
