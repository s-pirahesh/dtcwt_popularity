"""
Wikipedia Web Traffic converter
===============================
Converts the daily page views of the Kaggle competition "Web Traffic Time
Series Forecasting" (Google, 2017), file ``train_2.csv`` (or ``train_2.csv.zip``),
to the standard format ``timestamp,item_id,count``.

The raw file is wide: one row per page (column ``Page``, the name ends with
``_<project>_<access>_<agent>``) and one column per day from 2015-07-01 to
2017-09-10 (803 days, 145,063 rows).  The rules, all fixed before the data
were opened:

  shape check   the file must have the expected number of rows, consecutive
                daily columns from the first to the last date, and only
                non-negative integer values (or empty cells); otherwise nothing
                is written
  items         only the rows whose page name ends with ``_<access_agent>``
                (default ``all-access_all-agents``): every article of every
                project once; the other rows (desktop, mobile-web, spider) are
                parts of the same traffic
  subset        of the selected rows, in file order, the rows 0, k, 2k, ... are
                kept; k is the smallest integer with ceil(n / k) * days <= max_cells
                (n = number of selected rows).  The rule uses only the position
                of a row, never its values
  missing       an empty cell is zero views; as in the standard format, a day
                with zero views has no row
  item_id       the row number of the page in the raw file (from 0); the page
                names go to ``<output stem>_pages.csv``
  metadata      ``<output stem>_meta.json``: file hash, counts, k and the rules

Only counts are printed, never view values.

    python prepare_data.py --dataset wikipedia --input data/raw/wikipedia/train_2.csv.zip ^
           --output data/datasets/wikipedia_daily.csv
"""
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .base_converter import BaseConverter, ConverterFactory


def stride_for(n_rows: int, n_days: int, max_cells: int) -> int:
    """Smallest k >= 1 with ceil(n_rows / k) * n_days <= max_cells."""
    if n_rows < 1 or n_days < 1:
        raise ValueError('no rows or no days')
    if n_days > max_cells:
        raise ValueError(f'{n_days} days exceed max_cells = {max_cells}')
    k = 1
    while math.ceil(n_rows / k) * n_days > max_cells:
        k += 1
    return k


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class WikipediaConverter(BaseConverter):
    """Kaggle Wikipedia Web Traffic, daily page views."""

    DATASET_NAME = 'wikipedia'
    SUPPORTED_FILE_TYPES = ['.csv', '.zip']
    DESCRIPTION = 'Kaggle Wikipedia Web Traffic converter (daily page views, train_2.csv)'

    EXPECTED_ROWS = 145063
    FIRST_DATE = '2015-07-01'
    LAST_DATE = '2017-09-10'
    MAX_CELLS = 9256350     # rows scored by WSPI in the largest development scenario

    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'access_agent': {
                'type': str,
                'default': 'all-access_all-agents',
                'help': 'keep the pages whose name ends with _<access_agent>',
            },
            'max_cells': {
                'type': int,
                'default': cls.MAX_CELLS,
                'help': 'k is the smallest integer with ceil(n / k) * days <= max_cells',
            },
        }

    def __init__(self, access_agent: str = 'all-access_all-agents',
                 max_cells: int = MAX_CELLS, expected_rows: Optional[int] = EXPECTED_ROWS,
                 first_date: str = FIRST_DATE, last_date: str = LAST_DATE, **kwargs):
        super().__init__(**kwargs)
        self.access_agent = access_agent
        self.max_cells = int(max_cells)
        self.expected_rows = expected_rows
        self.first_date = pd.Timestamp(first_date)
        self.last_date = pd.Timestamp(last_date)
        self.info: Dict[str, Any] = {}
        self.pages: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def check_shape(self, raw: pd.DataFrame):
        """Dates (consecutive, from first to last) and row count.  Raises
        ValueError on any mismatch."""
        if raw.columns[0] != 'Page':
            raise ValueError(f'first column is {raw.columns[0]!r}, not Page')
        try:
            dates = pd.to_datetime(pd.Index(raw.columns[1:]), format='%Y-%m-%d')
        except (ValueError, TypeError) as e:
            raise ValueError(f'day columns are not dates: {e}')
        expected = pd.date_range(self.first_date, self.last_date, freq='D')
        if len(dates) != len(expected) or not (dates == expected).all():
            raise ValueError(f'day columns: {len(dates)} from {dates.min().date()} to '
                             f'{dates.max().date()}; expected {len(expected)} consecutive days '
                             f'from {self.first_date.date()} to {self.last_date.date()}')
        if self.expected_rows is not None and len(raw) != self.expected_rows:
            raise ValueError(f'{len(raw)} rows; expected {self.expected_rows}')
        if raw['Page'].isnull().any() or raw['Page'].duplicated().any():
            raise ValueError('empty or repeated page names')
        return dates

    def _convert_single_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        self.log(f'Reading {file_path.name}')
        raw = pd.read_csv(file_path, dtype={'Page': str})
        dates = self.check_shape(raw)
        values = raw.iloc[:, 1:].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if (values[finite] < 0).any():
            raise ValueError('negative view counts')
        if (values[finite] != np.round(values[finite])).any():
            raise ValueError('view counts that are not integers')
        if np.isinf(values).any():
            raise ValueError('infinite view counts')

        suffix = '_' + self.access_agent
        selected = np.flatnonzero(raw['Page'].str.endswith(suffix).to_numpy())
        if len(selected) == 0:
            raise ValueError(f'no page name ends with {suffix}')
        k = stride_for(len(selected), len(dates), self.max_cells)
        keep = selected[::k]

        v = values[keep]
        n_missing = int(np.isnan(v).sum())
        v = np.nan_to_num(v, nan=0.0).astype(np.int64)
        day_idx, item_pos = np.nonzero(v.T > 0)          # day-major, items ascending
        df = pd.DataFrame({'timestamp': dates[day_idx],
                           'item_id': keep[item_pos].astype(np.int64),
                           'count': v[item_pos, day_idx]})
        self.pages = pd.DataFrame({'item_id': keep.astype(np.int64),
                                   'page': raw['Page'].to_numpy()[keep]})
        self.info = dict(
            source_file=file_path.name, source_sha256=file_sha256(file_path),
            n_rows=int(len(raw)), n_days=int(len(dates)),
            first_date=str(dates[0].date()), last_date=str(dates[-1].date()),
            access_agent=self.access_agent, n_selected=int(len(selected)),
            max_cells=self.max_cells, stride_k=int(k), n_items=int(len(keep)),
            n_missing_cells_set_to_zero=n_missing,
            n_items_without_views=int((v.sum(axis=1) == 0).sum()),
            n_output_rows=int(len(df)),
            rules=dict(items=f'page name ends with {suffix}',
                       subset='rows 0, k, 2k, ... of the selected rows in file order; '
                              'k = smallest integer with ceil(n_selected / k) * n_days <= max_cells',
                       missing='empty cell = 0 views; days with 0 views have no row',
                       item_id='row number of the page in the raw file, from 0'))
        self.log(f'  rows {len(raw):,}, days {len(dates)} ({dates[0].date()} to {dates[-1].date()})')
        self.log(f'  selected ({suffix}): {len(selected):,}; k = {k}; items kept: {len(keep):,}')
        self.log(f'  output rows: {len(df):,}')
        return df

    def convert(self, input_path: Union[str, Path], output_path: Union[str, Path],
                **kwargs) -> pd.DataFrame:
        if isinstance(input_path, (list, tuple)):
            if len(input_path) != 1:
                raise ValueError('give exactly one raw file (train_2.csv or train_2.csv.zip)')
            input_path = input_path[0]
        df = super().convert(input_path, output_path, **kwargs)
        out = Path(output_path)
        pages = out.with_name(out.stem + '_pages.csv')
        meta = out.with_name(out.stem + '_meta.json')
        self.pages.to_csv(pages, index=False, encoding='utf-8')
        meta.write_text(json.dumps(dict(self.info, output=out.name, pages=pages.name),
                                   indent=2, ensure_ascii=False), encoding='utf-8')
        self.log(f'Pages: {pages}')
        self.log(f'Metadata: {meta}')
        return df

    def _save_output(self, df: pd.DataFrame, output_path: Path):
        if self.output_format == 'csv':
            df.to_csv(output_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        else:
            super()._save_output(df, output_path)


ConverterFactory.register('wikipedia', WikipediaConverter)
