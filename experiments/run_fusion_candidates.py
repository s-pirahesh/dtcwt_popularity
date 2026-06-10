r"""
WSPI Fusion-Candidate Comparison (wrapper-style, auto-extract)
==============================================================
Tests the variance-equalized fusion formulas (methods/wspi_fusion.py,
formal_analysis_formulation.md) as NEW methods, alongside the standard
methods and the incumbent WSPI. Pipeline + per-item API untouched.

Candidates:
    WSPI-Z2   z-fusion, {R, WE},        lam=1.0
    WSPI-Z3   z-fusion, {rho1, R, WE},  lam=1.0
    WSPI-Q2   quantile, {R, WE},        lam=1.0
    WSPI-Q3   quantile, {rho1, R, WE},  lam=1.0
    WSPI-Z2s  z-fusion, {R, WE},        lam=0.3
    WSPI-Z3s  z-fusion, {rho1, R, WE},  lam=0.3
    WSPI-2    multiplicative reference  (mu_L * exp(R - WE))

Respects --methods: only requested candidates are built, and calibration is
skipped entirely when no z/quantile candidate is requested (so this same
script is cheap to use for single standard-method runs under the parallel
orchestrator).

Usage (Windows):
    python experiments\run_fusion_candidates.py youtube
    python experiments\run_fusion_candidates.py youtube --methods WSPI-Z2 --incremental
    python experiments\run_fusion_candidates.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._auto_extract import fence, auto_extract
fence()

import numpy as np
from methods.wspi_fusion import WSPIFusion
from methods.wspi_candidates import WSPI2
import experiments.run_popularity_assessment as runner

# ---- CLI peeking (without disturbing runner's own argparse) ----------------
_dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'youtube'
_data_path = sys.argv[sys.argv.index('--data-path') + 1] if '--data-path' in sys.argv else None


def _requested_methods():
    if '--methods' not in sys.argv:
        return None                      # None => all
    out, i = [], sys.argv.index('--methods') + 1
    while i < len(sys.argv) and not sys.argv[i].startswith('--'):
        out.append(sys.argv[i]); i += 1
    return out


# Candidate factories (built lazily so calibration only happens if needed)
_ALL_CANDS = ('WSPI-Z2', 'WSPI-Z3', 'WSPI-Q2', 'WSPI-Q3',
              'WSPI-Z2s', 'WSPI-Z3s', 'WSPI-2')
_req = _requested_methods()
_want = [c for c in _ALL_CANDS if (_req is None or c in _req)]
_need_calib = any(c.startswith('WSPI-Z') or c.startswith('WSPI-Q') for c in _want)


def _calibration_series(max_items=2000):
    loader = runner.get_data_loader(_dataset, data_path=_data_path)
    df = loader.load_data()
    item_col = 'item_id'  if 'item_id'  in df.columns else getattr(loader, 'item_col', 'item_id')
    cnt_col  = 'count'    if 'count'    in df.columns else getattr(loader, 'count_col', 'count')
    time_col = 'timestamp' if 'timestamp' in df.columns else getattr(loader, 'time_col', None)
    totals = df.groupby(item_col)[cnt_col].sum().sort_values(ascending=False)
    series = []
    for item in totals.index[:max_items]:
        sub = df[df[item_col] == item]
        if time_col and time_col in sub.columns:
            sub = sub.sort_values(time_col)
        series.append(sub[cnt_col].to_numpy(dtype=np.float64))
    return series


_norm_z = _norm_q = None
if _want and _need_calib:
    print('=' * 70)
    print('FUSION-CANDIDATE COMPARISON  (calibrating normalizers, one pass) ...')
    _series = _calibration_series()
    _norm_z = WSPIFusion.calibrate(_series, mode='z')
    _norm_q = WSPIFusion.calibrate(_series, mode='quantile')
    print(f'  calibrated on {len(_series)} items | z-std logmu={_norm_z.std["logmu"]:.3f} '
          f'R={_norm_z.std["R"]:.3f} WE={_norm_z.std["WE"]:.3f} rho1={_norm_z.std["rho1"]:.3f}')
    print('=' * 70)


def _factory(name):
    return {
        'WSPI-Z2':  lambda: WSPIFusion('z',        False, 1.0, normalizer=_norm_z, name='WSPI-Z2'),
        'WSPI-Z3':  lambda: WSPIFusion('z',        True,  1.0, normalizer=_norm_z, name='WSPI-Z3'),
        'WSPI-Q2':  lambda: WSPIFusion('quantile', False, 1.0, normalizer=_norm_q, name='WSPI-Q2'),
        'WSPI-Q3':  lambda: WSPIFusion('quantile', True,  1.0, normalizer=_norm_q, name='WSPI-Q3'),
        'WSPI-Z2s': lambda: WSPIFusion('z',        False, 0.3, normalizer=_norm_z, name='WSPI-Z2s'),
        'WSPI-Z3s': lambda: WSPIFusion('z',        True,  0.3, normalizer=_norm_z, name='WSPI-Z3s'),
        'WSPI-2':   lambda: WSPI2(alpha=1.0, beta=1.0, name='WSPI-2'),
    }[name]()


import evaluation.method_configs as mc
from evaluation.method_configs import MethodConfig
for cname in _want:
    mc.METHOD_CONFIGS[cname] = MethodConfig(
        name=cname, window_slots=64, min_observations=32,
        description=f'Fusion candidate: {cname}')

_original_create_methods = runner.create_methods_dict


def _patched_create_methods(config):
    methods = _original_create_methods(config)          # standard, already --methods filtered
    req = set(config.methods) if config.methods else None
    for cname in _want:
        if req is None or cname in req:                 # respect --methods for candidates too
            methods[cname] = _factory(cname)
    return methods


runner.create_methods_dict = _patched_create_methods

if __name__ == '__main__':
    try:
        runner.main()
    finally:
        auto_extract('fusion')
