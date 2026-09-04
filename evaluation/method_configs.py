"""
Method-specific configurations for the Frozen Evaluation Protocol.

Five groups of popularity assessment methods:
  Group 1 — General Baselines:    AF, EWMA, RRD, VSE, CompoundPop, PFRF         (7-slot window)
  Group 2 — DWT Model:            DWT+AF                                       (64-slot window)
  Group 3 — DTCWT Model:          DTCWT+AF                                     (64-slot window)
  Group 4 — WSPI (Proposed):      WSPI                                         (64-slot window)
  Group 5 — Explicit Forecasters: Persistence, Holt, ARYW, ARIMA               (defense-gap experiment)

Note:
  - LRU removed: it is a cache eviction policy, not a popularity metric.
  - LFU renamed to MeanFreq: clearer description of the actual formula used.
  - New baselines RRD, VSE, CompoundPop added from literature.
  - Group 5 added for the explicit value-forecasting comparison
    (ARMA sliding-window approach of Hassine et al. 2017 [25] + naive/
    smoothing forecasters). These predict future demand values and are
    scored by the same future-ground-truth protocol.

Author: Sajjad
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class MethodConfig:
    """Configuration for a single assessment method."""
    name: str
    window_slots: int       # training window length in SLOTS (dataset-agnostic)
    min_observations: int   # minimum time-series length required
    description: str


METHOD_CONFIGS: Dict[str, MethodConfig] = {

    # =========================================================================
    # Group 1: General Popularity Baselines (7-slot window)
    # All are general-purpose popularity metrics, not cache-specific.
    # =========================================================================
    'AF': MethodConfig(
        name='AF',
        window_slots=7,
        min_observations=3,
        description='Access Frequency — exponentially decayed request sum [Chang & Chang, 2008]'
    ),

    'EWMA': MethodConfig(
        name='EWMA',
        window_slots=7,
        min_observations=3,
        description='Exponentially Weighted Moving Average (alpha=0.2) [Gui & Chen, 2020]'
    ),

    'RRD': MethodConfig(
        name='RRD',
        window_slots=7,
        min_observations=3,
        description='Requests/Lifetime ratio — normalises frequency by data age [Al Mistarihi & Yong, 2008]'
    ),

    'VSE': MethodConfig(
        name='VSE',
        window_slots=7,
        min_observations=3,
        description='Volume + Recency combined score [Mansouri & Asadi, 2014]'
    ),

    'CompoundPop': MethodConfig(
        name='CompoundPop',
        window_slots=7,
        min_observations=3,
        description='Three-factor compound popularity: total + recent + recency [Ye et al., 2014]'
    ),

    'PFRF': MethodConfig(
        name='PFRF',
        window_slots=7,
        min_observations=3,
        description='Period-based popularity weight — boost active periods, decay inactive [Lee et al., 2012]'
    ),

    # =========================================================================
    # Group 2: Trend-Shock Model — DWT  (Chapter 3, Section 3-2)
    # =========================================================================
    'DWT+AF': MethodConfig(
        name='DWT+AF',
        window_slots=64,
        min_observations=32,
        description=(
            'Trend-Shock Model (Section 3-2): '
            'Score = WAF(cA_L) + beta*WAF(cD_1), shift-sensitive'
        )
    ),

    # =========================================================================
    # Group 3: Stable DTCWT Model  (Chapter 3, Section 3-3)
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

    # =========================================================================
    # Group 5: Explicit Value-Forecasting Baselines (defense-gap experiment)
    # These FORECAST next-horizon demand; the score is the sum of forecasts.
    # Evaluated with the exact same future-ground-truth protocol.
    # =========================================================================
    'Persistence': MethodConfig(
        name='Persistence',
        window_slots=7,
        min_observations=3,
        description='Naive persistence forecast: next horizon repeats last observation'
    ),

    'Holt': MethodConfig(
        name='Holt',
        window_slots=64,
        min_observations=8,
        description='Holt double exponential smoothing (level+trend), h-step forecast sum'
    ),

    'ARYW': MethodConfig(
        name='ARYW',
        window_slots=64,
        min_observations=32,
        description='AR(2) fitted by Yule-Walker (closed-form), recursive h-step forecast'
    ),

    'ARIMA': MethodConfig(
        name='ARIMA',
        window_slots=64,
        min_observations=32,
        description=(
            'Sliding-window ARMA(1,1) via statsmodels MLE '
            '[Hassine et al., 2017 — proposal ref 25]; SLOW, run selectively'
        )
    ),

    # =========================================================================
    # Group 6: WSPI-F — Forecasted WSPI (Level-2 module)
    # Predicts DTCWT trend-band coefficients (persistence-anchored NLMS /
    # Yule-Walker AR) and scores the FUTURE window. Opt-in via --methods.
    # =========================================================================
    'WSPI-F': MethodConfig(
        name='WSPI-F',
        window_slots=64,
        min_observations=32,
        description=(
            'Forecasted WSPI: persistence-anchored NLMS on DTCWT trend-band '
            'coefficient increments (RCCWLMK family), score from extended coeffs'
        )
    ),

    'WSPI-F-YW': MethodConfig(
        name='WSPI-F-YW',
        window_slots=64,
        min_observations=32,
        description=(
            'Forecasted WSPI, Yule-Walker AR(2) coefficient predictor '
            '(closed-form, non-adaptive control variant)'
        )
    ),

    # =========================================================================
    # Group 7: WSPI-F2 / WSPI-FT — second-generation Level-2 module.
    # Same window and min_observations as WSPI so the window sets align
    # exactly with the cached WSPI / WSPI-F runs (do NOT change these two
    # numbers: a mismatch silently makes the comparison unfair).
    # Opt-in via --methods. See methods/wspi_forecast2.py for the diagnosis
    # that motivated them and for the provable inflation bound.
    # =========================================================================
    'WSPI-F2': MethodConfig(
        name='WSPI-F2',
        window_slots=64,
        min_observations=32,
        description=(
            'Gated forecasted WSPI: AR(2) Yule-Walker coefficient predictor '
            'with multiplicative clamp (c=4) and structural gate '
            'g=(R*(1-WE))^0.3; geometric blend, reduces EXACTLY to WSPI at g=0'
        )
    ),

    'WSPI-FT': MethodConfig(
        name='WSPI-FT',
        window_slots=64,
        min_observations=32,
        description=(
            'Robust forecasted WSPI: Theil-Sen median-of-slopes regression on '
            'the DTCWT trend-band coefficients with multiplicative clamp (c=4); '
            '29% breakdown point, so one burst coefficient cannot drag the trend'
        )
    ),
}


def get_method_config(method_name: str) -> MethodConfig:
    """Return MethodConfig for a given method name."""
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
    """Validate all METHOD_CONFIGS entries."""
    errors = []
    allowed_values = [7] + [2 ** i for i in range(3, 10)]

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


validate_configs()


if __name__ == '__main__':
    print("\nMethod configs (Chapter 3 alignment):\n")
    for name, cfg in METHOD_CONFIGS.items():
        print(f"  {name:<20} window={cfg.window_slots:>3} slots  min_obs={cfg.min_observations:>3}  |  {cfg.description}")
