r"""
WSPI Ablation Study (wrapper-style)
====================================
Runs five ablation variants of WSPI through the SAME pipeline that
run_popularity_assessment.py uses.  No reimplementation, no parallel
codepath — the variants are injected into create_methods_dict() and
registered in METHOD_CONFIGS before main() runs.

Variants (see methods/wspi_ablation.py):
    1. WSPI-noWE     (gamma = 0)              — drops wavelet-entropy penalty
    2. WSPI-noR      (beta  = 0)              — drops energy-ratio reward
    3. WSPI-noSL     (alpha = 0)              — drops trend-slope term
    4. WSPI-DWT      (DTCWT -> DWT)           — same features on DWT coeffs
    5. WSPI-noClip   (no exponent clamp)      — exponent unbounded

Full WSPI is always included for reference.

Output: results land in the standard timestamped folder
        results/<dataset>/w64_h7_..._<TIMESTAMP>/protocol/<variant>_protocol.parquet
        Use extract_fair_window.py to summarise them later.

Usage (Windows):
    python experiments\run_ablation.py youtube
    python experiments\run_ablation.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
# 4) Announce what we're doing and hand off to the standard main()
# ---------------------------------------------------------------------------
print('=' * 70)
print('ABLATION STUDY MODE')
print('=' * 70)
print('Standard 9 methods + 5 ablation variants:')
for vname in VARIANTS:
    print(f'  {vname:<14} window=64, params={VARIANTS[vname]}')
print('=' * 70)


if __name__ == '__main__':
    runner.main()
