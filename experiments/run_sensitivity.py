r"""
WSPI Sensitivity Analysis (wrapper-style)
==========================================
One-at-a-Time (OAT) sweep over each WSPI hyperparameter:
    alpha (slope weight)        in {0.25, 0.5, 1.0, 1.5, 2.0}   default = 1.0
    beta  (energy-ratio weight) in {0.0, 0.25, 0.5, 0.75, 1.0}  default = 0.5
    gamma (entropy weight)      in {0.0, 0.25, 0.5, 0.75, 1.0}  default = 0.5
    c     (clip bound)          in {1.0, 2.0, 3.0, 4.0, 5.0}    default = 3.0

Each parameter is varied one-at-a-time while the others stay at their default
values.  This yields ~20 variants per dataset, all run through the standard
pipeline.

Output: standard timestamped folder; use extract_fair_window.py to summarise.

Usage (Windows):
    python experiments\run_sensitivity.py youtube
    python experiments\run_sensitivity.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv

Tip: this run produces many variants (20+) and will take several hours.
For a quick check, edit the SWEEPS dict to use fewer values, or just run
the alpha and c sweeps first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.wspi_ablation import WSPIAblation
import evaluation.method_configs as mc
from evaluation.method_configs import MethodConfig
import experiments.run_popularity_assessment as runner

# ---------------------------------------------------------------------------
# 1) Define the OAT sweeps
# ---------------------------------------------------------------------------
SWEEPS = {
    'alpha': [0.25, 0.5, 1.0, 1.5, 2.0],
    'beta' : [0.0, 0.25, 0.5, 0.75, 1.0],
    'gamma': [0.0, 0.25, 0.5, 0.75, 1.0],
    'c'    : [1.0, 2.0, 3.0, 4.0, 5.0],
}
DEFAULT = dict(alpha=1.0, beta=0.5, gamma=0.5, c=3.0)


def _build_variants():
    """Materialise all OAT variants as (name, kwargs) pairs."""
    variants = {}
    for pname, values in SWEEPS.items():
        for v in values:
            kw = dict(DEFAULT)
            kw[pname] = v
            # Skip the default point if it would duplicate another sweep's default
            if kw == DEFAULT and 'WSPI-default' in variants:
                continue
            # Use a short, file-system-friendly name
            tag  = f'{pname}{str(v).replace(".", "p")}'
            name = f'WSPI-{tag}'
            variants[name] = dict(
                alpha_slope   = kw['alpha'],
                beta_ratio    = kw['beta'],
                gamma_entropy = kw['gamma'],
                clip_c        = kw['c'],          # WSPIAblation supports clip_c
                use_dtcwt     = True,
                use_clip      = True,
            )
    return variants


VARIANT_KW = _build_variants()


# ---------------------------------------------------------------------------
# 2) Register each variant in METHOD_CONFIGS
# ---------------------------------------------------------------------------
for vname in VARIANT_KW:
    mc.METHOD_CONFIGS[vname] = MethodConfig(
        name=vname,
        window_slots=64,
        min_observations=32,
        description=f'Sensitivity sweep variant: {vname}',
    )


# ---------------------------------------------------------------------------
# 3) Monkey-patch create_methods_dict to add the variants
# ---------------------------------------------------------------------------
_original_create_methods = runner.create_methods_dict


def _patched_create_methods(config):
    methods = _original_create_methods(config)
    for vname, kw in VARIANT_KW.items():
        methods[vname] = WSPIAblation(variant_name=vname, **kw)
    return methods


runner.create_methods_dict = _patched_create_methods

# ---------------------------------------------------------------------------
# 4) Announce and run
# ---------------------------------------------------------------------------
print('=' * 70)
print('SENSITIVITY ANALYSIS MODE')
print('=' * 70)
print(f'Standard 9 methods + {len(VARIANT_KW)} sensitivity variants:')
for pname, vals in SWEEPS.items():
    print(f'  sweep over {pname:<6} : {vals}')
print('=' * 70)


if __name__ == '__main__':
    runner.main()
