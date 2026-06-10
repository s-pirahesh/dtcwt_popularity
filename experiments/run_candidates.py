r"""
WSPI Candidate Comparison (wrapper-style, auto-extract)
=======================================================
Runs the TWO new candidate indices (WSPI-2, WSPI-3) alongside the 9
standard methods AND the current WSPI, all in a SINGLE evaluator run
that produces ONE results folder.  Nothing is replaced: the incumbent
WSPI is still evaluated, so you get a direct head-to-head.

After the run finishes, a tidy CSV summary is produced at:
    results/tables/candidates_<dataset>_<TIMESTAMP>.csv

Candidates (see methods/wspi_candidates.py):
    WSPI-2 : mu_L * exp(alpha*R - beta*WE)              alpha=1, beta=1
    WSPI-3 : mu_L * exp(alpha*Sm + beta*R - gamma*WE)   alpha=beta=gamma=1

Usage (Windows):
    python experiments\run_candidates.py youtube
    python experiments\run_candidates.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Record the fence BEFORE any heavy import.
from experiments._auto_extract import fence, auto_extract
fence()

# ---------------------------------------------------------------------------
# 1) Define the candidate methods
# ---------------------------------------------------------------------------
from methods.wspi_candidates import WSPI2, WSPI3

CANDIDATES = {
    'WSPI-2': lambda: WSPI2(alpha=1.0, beta=1.0, name='WSPI-2'),
    'WSPI-3': lambda: WSPI3(alpha=1.0, beta=1.0, gamma=1.0, name='WSPI-3'),
}

# ---------------------------------------------------------------------------
# 2) Register each candidate in METHOD_CONFIGS BEFORE the runner imports it
# ---------------------------------------------------------------------------
import evaluation.method_configs as mc
from evaluation.method_configs import MethodConfig

for cname in CANDIDATES:
    mc.METHOD_CONFIGS[cname] = MethodConfig(
        name=cname,
        window_slots=64,          # same window as full WSPI
        min_observations=32,      # same as full WSPI
        description=f'Candidate redesign of WSPI: {cname}',
    )

# ---------------------------------------------------------------------------
# 3) Monkey-patch create_methods_dict so the runner builds our candidates too
# ---------------------------------------------------------------------------
import experiments.run_popularity_assessment as runner

_original_create_methods = runner.create_methods_dict


def _patched_create_methods(config):
    """Build the standard methods (incl. WSPI), then add the candidates."""
    methods = _original_create_methods(config)
    for cname, factory in CANDIDATES.items():
        methods[cname] = factory()
    return methods


runner.create_methods_dict = _patched_create_methods

# ---------------------------------------------------------------------------
# 4) Announce, run, and auto-extract
# ---------------------------------------------------------------------------
print('=' * 70)
print('CANDIDATE COMPARISON MODE  (incumbent WSPI kept for head-to-head)')
print('=' * 70)
print('Standard 9 methods + WSPI + 2 candidates:')
print('  WSPI-2  : mu_L * exp(1*R - 1*WE)')
print('  WSPI-3  : mu_L * exp(1*Sm + 1*R - 1*WE)')
print('=' * 70)


if __name__ == '__main__':
    try:
        runner.main()
    finally:
        # ALWAYS try to produce a CSV, even if main() crashed late
        auto_extract('candidates')
