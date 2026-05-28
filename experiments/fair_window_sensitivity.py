r"""
Fair-Window Sensitivity (revised)
=================================
Runs the SAME evaluation as run_popularity_assessment.py, but with the
baseline methods (AF, EWMA, RRD, VSE, CompoundPop, PFRF) forced to use
a custom window size that you choose with --baseline-window.

The wavelet methods (DWT+AF, DTCWT+AF, WSPI) keep their original
window=64 because DTCWT with J=3 levels needs at least 16 samples and
cannot be evaluated on a window of 8 or below.

What this script does:
    1. Monkey-patches METHOD_CONFIGS so each baseline gets the chosen
       window before the runner reads it.
    2. Calls run_popularity_assessment.main() exactly once.
    3. Results land in the standard timestamped folder
       results/<dataset>/w<N>_h7_..._<TIMESTAMP>/
       (only one folder per run, no duplicates).

What this script does NOT do:
    - It does NOT run twice.
    - It does NOT touch the wavelet methods' window.
    - It does NOT write any extra files outside the standard pipeline.

Usage (Windows)
---------------
    REM Baselines with window = 8
    python experiments\fair_window_sensitivity.py youtube --baseline-window 8

    REM Baselines with window = 16
    python experiments\fair_window_sensitivity.py yellow_taxi ^
        --baseline-window 16 ^
        --data-path data\datasets\yellow_taxi_2025_all_hourly.csv

    REM Baselines with window = 64 (the original fair-window experiment)
    python experiments\fair_window_sensitivity.py youtube --baseline-window 64

You can pass through any other argument that run_popularity_assessment.py
accepts, e.g. --num-items, --data-path, --start-date, etc.

After the run, use extract_fair_window.py to summarise the results.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Allowed window sizes (must satisfy DWT/DTCWT validation: 7 or a power of 2)
ALLOWED_WINDOWS = {7, 8, 16, 32, 64, 128, 256}

BASELINE_NAMES = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']


def patch_baseline_window(target_window):
    """
    Replace every baseline's window_slots with target_window.
    Must be called BEFORE the runner imports METHOD_CONFIGS.
    """
    import evaluation.method_configs as mc
    from evaluation.method_configs import MethodConfig

    print('=' * 70)
    print(f'FAIR-WINDOW SENSITIVITY: baselines forced to window = {target_window}')
    print('=' * 70)

    # safe min_observations: at least 3, at most half of the window
    safe_min_obs = max(3, min(target_window // 2, 32))

    for name in BASELINE_NAMES:
        if name not in mc.METHOD_CONFIGS:
            continue
        cfg = mc.METHOD_CONFIGS[name]
        mc.METHOD_CONFIGS[name] = MethodConfig(
            name=cfg.name,
            window_slots=target_window,
            min_observations=safe_min_obs,
            description=cfg.description + f' [baseline_window={target_window}]'
        )

    print('Current method configs:')
    for nm, cfg in mc.METHOD_CONFIGS.items():
        print(f'  {nm:<15} window_slots = {cfg.window_slots:<3}  '
              f'min_obs = {cfg.min_observations}')
    print('=' * 70)


def parse_known_args():
    """
    Parse only our --baseline-window flag; everything else is passed
    through to the standard runner unchanged.
    """
    parser = argparse.ArgumentParser(
        add_help=False,
        description='Fair-window sensitivity wrapper.')
    parser.add_argument(
        '--baseline-window', type=int, required=True,
        help='Window size to force on baseline methods. '
             'Must be 7 or a power of 2 (e.g. 8, 16, 32, 64).'
    )
    own, passthrough = parser.parse_known_args()

    if own.baseline_window not in ALLOWED_WINDOWS:
        sys.stderr.write(
            'ERROR: --baseline-window must be one of '
            + str(sorted(ALLOWED_WINDOWS)) + '.\n'
            + '       You passed ' + str(own.baseline_window) + '.\n'
        )
        sys.exit(2)

    return own.baseline_window, passthrough


def main():
    target_window, passthrough = parse_known_args()

    # 1. Patch the configs FIRST.
    patch_baseline_window(target_window)

    # 2. Pretend the user invoked the standard runner directly.
    #    We rebuild sys.argv from the passthrough so its argparse sees a
    #    clean command line.
    sys.argv = [sys.argv[0]] + passthrough

    # 3. Import and run.  The runner reads the (now patched) configs
    #    when it builds the evaluation, so each baseline picks up its new
    #    window size automatically.
    from experiments.run_popularity_assessment import main as runner_main
    runner_main()


if __name__ == '__main__':
    main()
