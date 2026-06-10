r"""
WSPI Fusion-Candidate Comparison (wrapper-style, auto-extract)
==============================================================
Tests the variance-equalized fusion formulas (see methods/wspi_fusion.py and
formal_analysis_formulation.md) as NEW methods, alongside the 9 standard
methods, the incumbent WSPI, and WSPI-2 (the simple multiplicative reference).

Calibration: the dataset is loaded ONCE up-front to compute GLOBAL feature
normalizers (mean/std for z, empirical CDF for quantile). These are shared by
all fusion candidates. The per-item assess_single API and the evaluator are
left completely untouched.

Candidates injected:
    WSPI-Z2   z-fusion, {R, WE},        lam=1.0
    WSPI-Z3   z-fusion, {rho1, R, WE},  lam=1.0
    WSPI-Q2   quantile, {R, WE},        lam=1.0
    WSPI-Q3   quantile, {rho1, R, WE},  lam=1.0
    WSPI-Z2s  z-fusion, {R, WE},        lam=0.3   (soft: structure secondary)
    WSPI-Z3s  z-fusion, {rho1, R, WE},  lam=0.3   (soft)

Usage (Windows):
    python experiments\run_fusion_candidates.py youtube
    python experiments\run_fusion_candidates.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._auto_extract import fence, auto_extract
fence()

import numpy as np

# ---------------------------------------------------------------------------
# 1) One-time calibration: load data, build global normalizers
# ---------------------------------------------------------------------------
from methods.wspi_fusion import WSPIFusion
from methods.wspi_candidates import WSPI2
import experiments.run_popularity_assessment as runner

_dataset = sys.argv[1] if len(sys.argv) > 1 else 'youtube'
_data_path = None
if '--data-path' in sys.argv:
    _data_path = sys.argv[sys.argv.index('--data-path') + 1]


def _calibration_series(max_items: int = 2000):
    """Yield per-item count series (highest-volume first), for calibration."""
    loader = runner.get_data_loader(_dataset, data_path=_data_path)
    df = loader.load_data()

    # Resolve column names defensively (standard names first, loader attrs next)
    item_col = 'item_id'  if 'item_id'  in df.columns else getattr(loader, 'item_col', 'item_id')
    cnt_col  = 'count'    if 'count'    in df.columns else getattr(loader, 'count_col', 'count')
    time_col = 'timestamp' if 'timestamp' in df.columns else getattr(loader, 'time_col', None)

    # Rank items by total volume (mirror the 'top' selection used in eval)
    totals = df.groupby(item_col)[cnt_col].sum().sort_values(ascending=False)
    series = []
    for item in totals.index[:max_items]:
        sub = df[df[item_col] == item]
        if time_col and time_col in sub.columns:
            sub = sub.sort_values(time_col)
        series.append(sub[cnt_col].to_numpy(dtype=np.float64))
    return series


print('=' * 70)
print('FUSION-CANDIDATE COMPARISON MODE')
print('Calibrating global feature normalizers (one pass over the data) ...')
_series = _calibration_series()
_norm_z = WSPIFusion.calibrate(_series, mode='z')
_norm_q = WSPIFusion.calibrate(_series, mode='quantile')
print(f'  calibrated on {len(_series)} items')
print(f'  z-std  logmu={_norm_z.std["logmu"]:.3f}  R={_norm_z.std["R"]:.3f}  '
      f'WE={_norm_z.std["WE"]:.3f}  rho1={_norm_z.std["rho1"]:.3f}')
print('=' * 70)

# ---------------------------------------------------------------------------
# 2) Define candidates (factories) and register configs
# ---------------------------------------------------------------------------
CANDIDATES = {
    'WSPI-Z2':  lambda: WSPIFusion('z',        False, 1.0, normalizer=_norm_z, name='WSPI-Z2'),
    'WSPI-Z3':  lambda: WSPIFusion('z',        True,  1.0, normalizer=_norm_z, name='WSPI-Z3'),
    'WSPI-Q2':  lambda: WSPIFusion('quantile', False, 1.0, normalizer=_norm_q, name='WSPI-Q2'),
    'WSPI-Q3':  lambda: WSPIFusion('quantile', True,  1.0, normalizer=_norm_q, name='WSPI-Q3'),
    'WSPI-Z2s': lambda: WSPIFusion('z',        False, 0.3, normalizer=_norm_z, name='WSPI-Z2s'),
    'WSPI-Z3s': lambda: WSPIFusion('z',        True,  0.3, normalizer=_norm_z, name='WSPI-Z3s'),
    'WSPI-2':   lambda: WSPI2(alpha=1.0, beta=1.0, name='WSPI-2'),   # multiplicative reference
}

import evaluation.method_configs as mc
from evaluation.method_configs import MethodConfig

for cname in CANDIDATES:
    mc.METHOD_CONFIGS[cname] = MethodConfig(
        name=cname, window_slots=64, min_observations=32,
        description=f'Fusion candidate: {cname}',
    )

# ---------------------------------------------------------------------------
# 3) Monkey-patch create_methods_dict to inject candidates (WSPI kept)
# ---------------------------------------------------------------------------
_original_create_methods = runner.create_methods_dict


def _patched_create_methods(config):
    methods = _original_create_methods(config)
    for cname, factory in CANDIDATES.items():
        methods[cname] = factory()
    return methods


runner.create_methods_dict = _patched_create_methods

print('Standard 9 + WSPI + candidates:', list(CANDIDATES.keys()))
print('=' * 70)

# ---------------------------------------------------------------------------
# 4) Run + auto-extract
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        runner.main()
    finally:
        auto_extract('fusion')
