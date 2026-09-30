r"""
T1.1 — Audit of the REAL series length that each method receives
================================================================
Replays the window logic of IncrementalTemporalEvaluator exactly (through
the compat mode of evaluation/fast_evaluator.py, whose indexing is checked
against the V4 protocol CSVs in T1.3) and reports, per dataset and method:

  * the train slice length in slots (W-1 in the current code);
  * the distribution of the actual series length passed to assess_single
    (rows that exist in the CSV — missing slots are skipped, not zero-filled);
  * how many of those series have internal gaps (missing slots compressed);
  * the length after power-of-two padding (DTCWT+AF / WSPI) and the DWT
    level that DWTAssessment._safe_level() actually uses;
  * how often the eligible item set changes between two consecutive
    windows (relevant for RSI, which compares POSITIONAL top-K indices).

    python tools/audit_window_length.py --out results/revision_v5/audit_20260924
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastEvaluator, _dwt_safe_level

DATASETS = {
    # name: (csv, dataset min_observations used in the V4 run)
    'youtube': ('data/datasets/youtube_hourly.csv', 50),
    'yellow_taxi_hourly': ('data/datasets/yellow_taxi_2025_all_hourly.csv', 24),
}
METHODS = {  # name: (window_slots, min_obs) from evaluation/method_configs.py
    'baselines (AF..PFRF)': (7, 3),
    'DWT+AF': (64, 32),
    'DTCWT+AF': (64, 32),
    'WSPI': (64, 32),
}


def pad_len(L, level):
    return max(2 ** (level + 1), 2 ** int(np.ceil(np.log2(max(L, 2)))))


def audit(fe, W, min_obs, name):
    rows = []
    prev_set = None
    changed = 0
    n_win = 0
    for k in range(fe.num_slots + 1):
        lo, hi = fe.window_bounds(k, W)
        if fe.any_cum[hi] - fe.any_cum[lo] == 0 or not fe.any_row[k]:
            continue
        L = fe.C[:, hi] - fe.C[:, lo]
        idx = np.where(L >= min_obs)[0]
        if len(idx) < 2:
            continue
        n_win += 1
        Ls = L[idx]
        span = hi - lo
        # series with internal gaps: first..last present slot shorter than rows?
        cur = frozenset(idx.tolist())
        if prev_set is not None and cur != prev_set:
            changed += 1
        prev_set = cur
        rows.append(dict(window_id=k, slice_slots=span, n_items=len(idx),
                         len_min=int(Ls.min()), len_median=float(np.median(Ls)),
                         len_max=int(Ls.max()),
                         frac_full=float(np.mean(Ls == span)),
                         frac_missing_slots=float(np.mean(Ls < span))))
    df = pd.DataFrame(rows)
    return df, changed, n_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/revision_v5/audit_20260924')
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for ds, (csv, dmin) in DATASETS.items():
        fe = FastEvaluator.from_csv(ROOT / csv, dataset_min_obs=dmin, verbose=True)
        # timestamps present in the file vs. the regular grid
        grid = fe.num_slots + 1
        present_slots = int(fe.any_row[:grid].sum())
        for mname, (W, mo) in METHODS.items():
            df, changed, n_win = audit(fe, W, mo, mname)
            df.to_csv(out / f'window_lengths_{ds}_{mname.split(" ")[0]}.csv', index=False)
            steady = df[df['slice_slots'] == W - 1]
            allL = []
            for k in steady['window_id']:
                lo, hi = fe.window_bounds(int(k), W)
                L = fe.C[:, hi] - fe.C[:, lo]
                allL.append(L[L >= mo])
            allL = np.concatenate(allL) if allL else np.array([], int)
            s = dict(dataset=ds, method=mname, window_slots_config=W,
                     train_slice_slots=W - 1,
                     grid_slots=grid, slots_with_any_row=present_slots,
                     windows_scored=n_win,
                     windows_with_full_slice=int(len(steady)),
                     series_len_min=int(allL.min()) if len(allL) else None,
                     series_len_median=float(np.median(allL)) if len(allL) else None,
                     series_len_max=int(allL.max()) if len(allL) else None,
                     share_series_len_eq_W_minus_1=float(np.mean(allL == W - 1)) if len(allL) else None,
                     share_series_with_missing_slots=float(np.mean(allL < W - 1)) if len(allL) else None,
                     windows_item_set_changed=int(changed))
            if W >= 32 and len(allL):
                s['padded_len_values_J2'] = sorted({pad_len(int(x), 2) for x in np.unique(allL)})
                s['padded_len_values_J3'] = sorted({pad_len(int(x), 3) for x in np.unique(allL)})
                lv = np.array([_dwt_safe_level(int(x), 3) for x in allL])
                s['dwt_effective_level_share'] = {int(v): float(np.mean(lv == v)) for v in np.unique(lv)}
                s['share_needing_padding_to_64'] = float(np.mean(allL < 64))
            summary.append(s)
            print(json.dumps(s, default=str))
    (out / 'window_length_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    pd.DataFrame(summary).to_csv(out / 'window_length_summary.csv', index=False)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
