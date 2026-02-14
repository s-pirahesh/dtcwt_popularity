"""
Method-specific configurations for the Frozen Evaluation Protocol.

Three assessment models from Chapter 3:
  Group 1 — Baselines:    AF, LRU, LFU, EWMA        (7-day window)
  Group 2 — DWT Model:    DWT+AF                     (64-day window, Section 3-2)
  Group 3 — DTCWT Model:  DTCWT+AF                   (64-day window, Section 3-3)
  Group 4 — WSPI:         Proposed method             (64-day window, Section 3-4)

Note: 'Statistical' (skewness/kurtosis) has been removed.
      'Hybrid V3.0' and 'Hybrid V3.1' have been replaced by 'WSPI'.

Author: Sajjad
"""

from dataclasses import dataclass
from typing import Dict
import math


@dataclass
class MethodConfig:
    """Configuration for a single assessment method."""
    name: str
    window_slots: int       # training window length in SLOTS (dataset-agnostic: 7 for baselines, 64 for wavelet methods)
    min_observations: int   # minimum time-series length required
    description: str


# All method configurations aligned with Chapter 3 methodology
METHOD_CONFIGS: Dict[str, MethodConfig] = {

    # =========================================================================
    # Group 1: Baselines (7-day window)
    # Fast, responsive, low data requirement.
    # =========================================================================
    'AF': MethodConfig(
        name='AF',
        window_slots=7,
        min_observations=3,
        description='Access Frequency — count-based baseline'
    ),

    'LRU': MethodConfig(
        name='LRU',
        window_slots=7,
        min_observations=3,
        description='Least Recently Used — recency-based baseline'
    ),

    'LFU': MethodConfig(
        name='LFU',
        window_slots=7,
        min_observations=3,
        description='Least Frequently Used — frequency-based baseline'
    ),

    'EWMA': MethodConfig(
        name='EWMA',
        window_slots=7,
        min_observations=3,
        description='Exponentially Weighted Moving Average (alpha=0.3)'
    ),

    # =========================================================================
    # Group 2: Trend-Shock Model — DWT  (Chapter 3, Section 3-2)
    #
    # Motivation: popularity signals have two distinct components —
    #   cA_L : smooth long-term trend (gradual adoption)
    #   cD_1 : high-frequency burst   (viral shock)
    # Formula:
    #   Score_DWT = WAF(cA_L) + beta * WAF(cD_1)
    #   WAF(X)    = sum |X[t-i]| * 2^{-i}   (exponential decay weighting)
    #
    # Limitation: DWT is shift-sensitive — a small shift in the window
    # boundary changes coefficients noticeably (motivates DTCWT below).
    # =========================================================================
    'DWT+AF': MethodConfig(
        name='DWT+AF',
        window_slots=64,
        min_observations=32,
        description=(
            'Trend-Shock Model (Section 3-2): '
            'Score = WAF(cA_L) + beta*WAF(cD_1), shift-sensitive baseline'
        )
    ),

    # =========================================================================
    # Group 3: Stable DTCWT Model  (Chapter 3, Section 3-3)
    #
    # Dual-Tree Complex Wavelet Transform produces approximately
    # shift-invariant complex coefficients:
    #   psi_c(t) = psi_r(t) + j*psi_i(t)
    # Feature extraction:
    #   M_trend = |Low|    (magnitude of low-band complex coefficients)
    #   M_shock = |High_1| (magnitude of first high-band)
    # Formula:
    #   Score_DTCWT = WAF(M_trend) + beta * WAF(M_shock)
    #
    # Improvement over DWT: stable ranking under window shifts.
    # Remaining gap: energy-only, ignores relative growth and entropy
    # (motivates WSPI).
    # =========================================================================
    'DTCWT+AF': MethodConfig(
        name='DTCWT+AF',
        window_slots=64,
        min_observations=32,
        description=(
            'Stable DTCWT Model (Section 3-3): '
            'Score = WAF(M_trend) + beta*WAF(M_shock), shift-invariant'
        )
    ),

    # =========================================================================
    # Group 4: WSPI — Proposed Method  (Chapter 3, Section 3-4)
    #
    # Redefines popularity as a structural, multi-scale property:
    #   mu_L — trend volume   : WeightedMean(|Low|)  with 2^{-i} weights
    #   S_L  — normalised slope: Slope(|Low|) / (Mean(|Low|) + eps)
    #   R    — energy ratio   : E_low / (E_low + sum E_high)  [stability]
    #   WE   — wavelet entropy : -sum p_i*log2(p_i)            [disorder]
    #
    # Final formula:
    #   P_WSPI = mu_L * exp( clip( alpha*S_L + beta*R - gamma*WE, -3, 3 ) )
    #
    # Frozen parameters (Chapter 3 specification):
    #   alpha = 1.0   (trend slope weight)
    #   beta  = 0.5   (energy-ratio weight)
    #   gamma = 0.5   (wavelet entropy penalty)
    #
    # Clamp range [-3, 3] → multiplier in [exp(-3)~0.05, exp(3)~20]
    # Complexity: O(N) — linear in signal length.
    # =========================================================================
    'WSPI': MethodConfig(
        name='WSPI',
        window_slots=64,
        min_observations=32,
        description=(
            'Wavelet Structural Popularity Index (Section 3-4): '
            'P = mu_L * exp(clip(alpha*S_L + beta*R - gamma*WE, -3, 3)), '
            'alpha=1.0, beta=0.5, gamma=0.5'
        )
    ),
}


def get_method_config(method_name: str) -> MethodConfig:
    """
    Return the MethodConfig for a given method name.

    Raises:
        KeyError: if method_name is not registered.
    """
    if method_name not in METHOD_CONFIGS:
        raise KeyError(f"No config found for method: {method_name}")
    return METHOD_CONFIGS[method_name]


def get_window_size(method_name: str, default: int = 30) -> int:
    """Return window_slots for method_name, or default if not registered."""
    try:
        return get_method_config(method_name).window_slots
    except KeyError:
        return default


def get_min_observations(method_name: str, default: int = 10) -> int:
    """Return min_observations for method_name, or default if not registered."""
    try:
        return get_method_config(method_name).min_observations
    except KeyError:
        return default


def list_methods_by_window_size() -> Dict[int, list]:
    """Return {window_size: [method_names]} grouped by window_slots."""
    grouped: Dict[int, list] = {}
    for name, config in METHOD_CONFIGS.items():
        grouped.setdefault(config.window_slots, []).append(name)
    return grouped


def validate_configs():
    """
    Validate all METHOD_CONFIGS entries.
    window_slots must be 7 or a power of 2 (8..512).
    min_observations must be >= 1 and <= window_slots.
    """
    errors = []
    allowed_values = [7] + [2 ** i for i in range(3, 10)]  # 7, 8, 16, 32, 64, 128, 256, 512

    for name, config in METHOD_CONFIGS.items():
        if config.window_slots not in allowed_values:
            errors.append(
                f"{name}: window_slots={config.window_slots} must be 7 or a power of 2"
            )
        if config.min_observations < 1:
            errors.append(f"{name}: min_observations must be >= 1")
        if config.min_observations > config.window_slots:
            errors.append(
                f"{name}: min_observations={config.min_observations} "
                f"> window_slots={config.window_slots}"
            )

    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))

    print("=" * 70)
    print("METHOD CONFIGS VALIDATION")
    print("=" * 70)

    grouped = list_methods_by_window_size()
    for window_size in sorted(grouped.keys()):
        methods = grouped[window_size]
        print(f"\nWindow {window_size} slots ({len(methods)} methods):")
        for method in methods:
            cfg = METHOD_CONFIGS[method]
            print(f"  • {method:<20} min_obs={cfg.min_observations:>3}")

    print("\n" + "=" * 70)
    print(f"All {len(METHOD_CONFIGS)} method configs validated successfully")
    print("=" * 70 + "\n")


# Run validation on import
validate_configs()


if __name__ == '__main__':
    print("\nMethod configs (Chapter 3 alignment):\n")
    for name, cfg in METHOD_CONFIGS.items():
        print(f"  {name:<15} window={cfg.window_slots:>3} slots  min_obs={cfg.min_observations:>3}  |  {cfg.description}")
