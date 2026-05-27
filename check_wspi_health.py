"""
WSPI Health Check
=================
Runs WSPI on a known synthetic signal and checks whether the main
algorithm executed (or whether the silent fallback to np.mean() kicked in).

Background
----------
Newer versions of the `dtcwt` library (>= 0.13?) return `pyramid.lowpass`
as a 2-D array of shape (n, 1) instead of a 1-D array of shape (n,).
In `methods/hybrid_assessment.py` this causes:

    np.average(lowpass_mags, weights=weights)
    # -> ValueError: Axis must be specified when shapes of a and weights differ.

The exception is caught silently and assess_single() falls back to
`float(np.mean(time_series))` — meaning every WSPI call returns the raw
sample mean. This script detects that situation.

Usage
-----
    cd dtcwt_popularity
    python check_wspi_health.py

Exits 0 if WSPI is healthy, 1 if a fallback is being used.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import dtcwt
from config import WAVELET_CONFIG
from methods.hybrid_assessment import HybridAssessment


def check_dtcwt_output_shape():
    """Check whether dtcwt returns 1-D or 2-D lowpass."""
    print("=" * 70)
    print("STEP 1: Check dtcwt output shape")
    print("=" * 70)
    print(f"  dtcwt version:        {dtcwt.__version__}")

    signal = np.linspace(5, 15, 64).astype(np.float64)
    xform  = dtcwt.Transform1d(
        biort=WAVELET_CONFIG['dtcwt_biort'],
        qshift=WAVELET_CONFIG['dtcwt_qshift'],
    )
    pyr = xform.forward(signal, nlevels=3)
    print(f"  signal.shape:         {signal.shape}")
    print(f"  pyramid.lowpass.shape:{pyr.lowpass.shape}")
    print(f"  pyramid.lowpass.ndim: {pyr.lowpass.ndim}")

    if pyr.lowpass.ndim == 2:
        print()
        print("  >>> WARNING: dtcwt returns 2-D lowpass.")
        print("  >>> This will cause np.average() to fail and WSPI will")
        print("  >>> silently fall back to np.mean() of the raw signal.")
        return False
    print()
    print("  >>> OK: dtcwt returns 1-D lowpass.")
    return True


def check_wspi_executes():
    """Verify that WSPI's main algorithm runs (not the fallback)."""
    print()
    print("=" * 70)
    print("STEP 2: Check whether WSPI runs the main algorithm")
    print("=" * 70)

    # Build a signal where the raw mean is clearly different from WSPI's
    # structural answer (steep upward trend -> WSPI multiplies mu_L by
    # exp(positive z) -> score > raw mean).
    np.random.seed(42)
    n = 64
    trend  = np.linspace(5, 50, n)
    noise  = np.random.normal(0, 1, n)
    signal = np.maximum(0, trend + noise)

    raw_mean   = float(np.mean(signal))
    print(f"  raw mean(signal) = {raw_mean:.4f}")

    method     = HybridAssessment()
    wspi_score = method.assess_single(signal)
    print(f"  WSPI score       = {wspi_score:.4f}")

    if abs(wspi_score - raw_mean) < 1e-6:
        print()
        print("  >>> FAILURE: WSPI returned EXACTLY mean(signal).")
        print("  >>> This means the main algorithm threw an exception")
        print("  >>> and the silent fallback was used.")
        print("  >>> See section B.3 of three_findings_solutions.md for the fix.")
        return False
    print()
    print("  >>> OK: WSPI score differs from raw mean.")
    print("  >>> The main algorithm executed successfully.")
    return True


def check_directional_sensitivity():
    """Sanity-check: ascending signal should score higher than descending."""
    print()
    print("=" * 70)
    print("STEP 3: Directional sensitivity sanity check")
    print("=" * 70)

    np.random.seed(0)
    base    = np.linspace(5, 20, 64) + np.random.normal(0, 1, 64)
    ts_up   = np.maximum(0, base)
    ts_down = ts_up[::-1].copy()
    ts_flat = np.full(64, float(np.mean(ts_up)))

    m = HybridAssessment()
    su = m.assess_single(ts_up)
    sd = m.assess_single(ts_down)
    sf = m.assess_single(ts_flat)

    print(f"  ascending  signal : WSPI = {su:.4f}")
    print(f"  descending signal : WSPI = {sd:.4f}")
    print(f"  flat       signal : WSPI = {sf:.4f}")

    if su > sf > sd:
        print()
        print("  >>> OK: ordering matches the design (ascending > flat > descending).")
        return True
    print()
    print("  >>> WARNING: ordering does not match expectation.")
    print("  >>> The slope feature S_L may not be propagating to the final score.")
    return False


if __name__ == "__main__":
    s1 = check_dtcwt_output_shape()
    s2 = check_wspi_executes()
    s3 = check_directional_sensitivity()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Step 1 (dtcwt 1-D output) : {'OK' if s1 else 'INFO (2-D is OK if patch applied)'}")
    print(f"  Step 2 (no fallback)      : {'OK' if s2 else 'FAILED'}")
    print(f"  Step 3 (directional)      : {'OK' if s3 else 'WARN'}")
    print()
    if s2 and s3:
        if s1:
            print("  All checks passed. WSPI is operating correctly.")
            print("  Previously reported results are valid.")
        else:
            print("  WSPI runs correctly despite 2-D dtcwt output —")
            print("  this means the ravel() patch is applied.")
            print("  Previously reported results are valid (if patch was in place).")
        sys.exit(0)
    elif not s2:
        print("  CRITICAL: WSPI is currently broken (silent fallback to mean).")
        print("  Apply the patch in three_findings_solutions.md (section B.3)")
        print("  and re-run ALL experiments before submitting.")
        sys.exit(1)
    else:
        print("  WSPI runs but at least one sanity check is unusual.")
        print("  Worth investigating before relying on the numbers.")
        sys.exit(1)
