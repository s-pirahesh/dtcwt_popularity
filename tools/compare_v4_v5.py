r"""
T1.4 — V4 protocol vs V5 protocol, per method (all windows of each method)
==========================================================================
Reads the V4 comparison/main_summary.csv and the V5 summary_all_windows.csv
(+ the DTCWT+AF J=2 ablation under V5) and writes one tidy CSV.

    python tools/compare_v4_v5.py            (writes results/revision_v5/T1.4_protocol_v5/v4_vs_v5_all_windows.csv)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results'

RUNS = {
    'youtube_hourly': dict(
        v4=R / 'youtube/main_20260612_140555/comparison/main_summary.csv',
        v5=R / 'revision_v5/T1.4_protocol_v5/youtube_hourly/comparison/summary_all_windows.csv',
        j2=R / 'revision_v5/T1.4_protocol_v5/ablation_dtcwt_J2/youtube_hourly/comparison/summary_all_windows.csv'),
    'taxi_hourly': dict(
        v4=R / 'yellow_taxi/predcmp_20260830_001817_hourly/comparison/main_summary.csv',
        v5=R / 'revision_v5/T1.4_protocol_v5/taxi_hourly/comparison/summary_all_windows.csv',
        j2=R / 'revision_v5/T1.4_protocol_v5/ablation_dtcwt_J2/taxi_hourly/comparison/summary_all_windows.csv'),
    # V4 paper values for 30/5 min come from the main_20260620 runs (tex line 354: 0.573, 0.728)
    'taxi_30min': dict(
        v4=R / 'yellow_taxi/main_20260620_092517_30min/comparison/main_summary.csv',
        v5=R / 'revision_v5/T1.4_protocol_v5/taxi_30min/comparison/summary_all_windows.csv',
        j2=R / 'revision_v5/T1.4_protocol_v5/ablation_dtcwt_J2/taxi_30min/comparison/summary_all_windows.csv'),
    'taxi_5min': dict(
        v4=R / 'yellow_taxi/main_20260620_170044_5min/comparison/main_summary.csv',
        v5=R / 'revision_v5/T1.4_protocol_v5/taxi_5min/comparison/summary_all_windows.csv',
        j2=R / 'revision_v5/T1.4_protocol_v5/ablation_dtcwt_J2/taxi_5min/comparison/summary_all_windows.csv'),
}
METRICS = ['ndcg@10', 'coverage@10', 'kendall_tau', 'spearman_rho', 'rsi@10', 'robustness_distortion']
METHODS = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF', 'DWT+AF', 'DTCWT+AF', 'WSPI']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/revision_v5/T1.4_protocol_v5')
    a = ap.parse_args()
    rows = []
    for ds, p in RUNS.items():
        v4 = pd.read_csv(p['v4'], encoding='utf-8-sig').set_index('method')
        v5 = pd.read_csv(p['v5']).set_index('method')
        j2 = pd.read_csv(p['j2']).set_index('method') if p['j2'].exists() else None
        for m in METHODS:
            r = {'dataset': ds, 'method': m,
                 'windows_v4': int(v4.loc[m, 'windows']), 'windows_v5': int(v5.loc[m, 'n_windows'])}
            for c in METRICS:
                r[f'{c}_v4'] = v4.loc[m, c] if c in v4.columns else float('nan')
                r[f'{c}_v5'] = v5.loc[m, f'{c}_mean']
            if m == 'DTCWT+AF' and j2 is not None:
                for c in METRICS:
                    r[f'{c}_v5_J2'] = j2.loc[m, f'{c}_mean']
            r['ties_top21_share_v5'] = v5.loc[m, 'ties_top21_share']
            rows.append(r)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'v4_vs_v5_all_windows.csv', index=False, encoding='utf-8')
    pd.set_option('display.width', 250)
    print(df[['dataset', 'method', 'windows_v4', 'windows_v5', 'ndcg@10_v4', 'ndcg@10_v5',
              'rsi@10_v4', 'rsi@10_v5', 'robustness_distortion_v4',
              'robustness_distortion_v5']].round(4).to_string(index=False))
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    sys.exit(main())
