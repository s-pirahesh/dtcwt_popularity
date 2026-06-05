r"""
WSPI Ablation Study (wrapper-style, auto-extract)
==================================================
Runs five ablation variants of WSPI alongside the 9 standard methods,
all in a SINGLE evaluator run that produces ONE results folder.

After the run finishes, a tidy CSV summary is automatically produced at:
    results/tables/ablation_<dataset>_<TIMESTAMP>.csv

Variants (see methods/wspi_ablation.py):
    1. WSPI-noWE     (gamma = 0)              — drops wavelet-entropy penalty
    2. WSPI-noR      (beta  = 0)              — drops energy-ratio reward
    3. WSPI-noSL     (alpha = 0)              — drops trend-slope term
    4. WSPI-DWT      (DTCWT -> DWT)           — same features on DWT coeffs
    5. WSPI-noClip   (no exponent clamp)      — exponent unbounded

Usage (Windows):
    python experiments\run_ablation.py youtube
    python experiments\run_ablation.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Record the fence BEFORE any heavy import, so the auto-extractor knows the
# cut-off for "new folders created during this run".
from experiments._auto_extract import fence, auto_extract
fence()

# ---------------------------------------------------------------------------
# 1) Define the ablation variants
# ---------------------------------------------------------------------------
from methods.wspi_ablation import WSPIAblation

VARIANTS = {
    'WSPI-noWE':   dict(alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.0,
                        use_dtcwt=True,  use_clip=True),
    'WSPI-noR':    dict(alpha_slope=1.0, beta_ratio=0.0, gamma_entropy=0.5,
                        use_dtcwt=True,  use_clip=True),
    'WSPI-noSL':   dict(alpha_slope=0.0, beta_ratio=0.5, gamma_entropy=0.5,
                        use_dtcwt=True,  use_clip=True),
    'WSPI-DWT':    dict(alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.5,
                        use_dtcwt=False, use_clip=True),
    'WSPI-noClip': dict(alpha_slope=1.0, beta_ratio=0.5, gamma_entropy=0.5,
                        use_dtcwt=True,  use_clip=False),
}

# ---------------------------------------------------------------------------
# 2) Register each variant in METHOD_CONFIGS BEFORE the runner imports it
# ---------------------------------------------------------------------------
import evaluation.method_configs as mc
from evaluation.method_configs import MethodConfig

for vname in VARIANTS:
    mc.METHOD_CONFIGS[vname] = MethodConfig(
        name=vname,
        window_slots=64,                # same as full WSPI
        min_observations=32,            # same as full WSPI
        description=f'Ablation variant of WSPI: {vname}',
    )

# ---------------------------------------------------------------------------
# 3) Monkey-patch create_methods_dict so the runner builds our variants too
# ---------------------------------------------------------------------------
import experiments.run_popularity_assessment as runner

_original_create_methods = runner.create_methods_dict


def _patched_create_methods(config):
    """Build the standard 9 methods, then inject the ablation variants."""
    methods = _original_create_methods(config)
    for vname, cfg in VARIANTS.items():
        methods[vname] = WSPIAblation(variant_name=vname, **cfg)
    return methods


runner.create_methods_dict = _patched_create_methods

# ---------------------------------------------------------------------------
# 4) Announce, run, and auto-extract
# ---------------------------------------------------------------------------
print('=' * 70)
print('ABLATION STUDY MODE')
print('=' * 70)
print('Standard 9 methods + 5 ablation variants:')
for vname in VARIANTS:
    print(f'  {vname:<14} window=64, params={VARIANTS[vname]}')
print('=' * 70)


if __name__ == '__main__':
    try:
        runner.main()
    finally:
        # ALWAYS try to produce a CSV, even if main() crashed late
        auto_extract('ablation')
