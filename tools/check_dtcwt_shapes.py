r"""
T1.1 — Audit of DTCWT / DWT output shapes (revision-srep-v5)
============================================================
Prints and saves the shape and dtype of every sub-band produced by the
transforms used in this project, for N in {16, 32, 64, 128} and
J in {2, 3, 4, 5}.

  * DTCWT : dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
            -- the same object WSPIAssessment and DTCWTAssessment build.
  * DWT   : pywt.wavedec(x, 'db4', level=J, mode='symmetric')
            -- the same call DWTAssessment / WSPI(use_dtcwt=False) make.

It also records the *effective* DWT level that DWTAssessment._safe_level()
would pick for each N (it silently lowers J for short signals).

Usage (from the project root):
    python tools/check_dtcwt_shapes.py
    python tools/check_dtcwt_shapes.py --out results/revision_v5/audit_20260924

Nothing in the existing code is modified or imported for side effects,
except WAVELET_CONFIG (read only).
"""
import argparse
import math
import sys
from pathlib import Path

import warnings
import numpy as np
import pywt
import dtcwt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import WAVELET_CONFIG
    BIORT = WAVELET_CONFIG['dtcwt_biort']
    QSHIFT = WAVELET_CONFIG['dtcwt_qshift']
    DWT_WAVELET = WAVELET_CONFIG['dwt_wavelet']
except Exception:                       # stand-alone fallback
    BIORT, QSHIFT, DWT_WAVELET = 'near_sym_a', 'qshift_a', 'db4'

N_LIST = [16, 32, 64, 128]
J_LIST = [2, 3, 4, 5]


def dwt_safe_level(n, level, wavelet=DWT_WAVELET):
    """Exact copy of DWTAssessment._safe_level (read-only replica)."""
    filter_len = pywt.Wavelet(wavelet).dec_len
    if filter_len <= 1 or n <= 0:
        return 1
    max_safe = int(math.floor(math.log2(n / (filter_len - 1)))) if n >= filter_len else 1
    return max(1, min(level, max_safe))


def fmt(a):
    a = np.asarray(a)
    return f"{tuple(a.shape)} {a.dtype}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None, help='output folder for the CSV report')
    args = ap.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning, module='pywt')
    rng = np.random.RandomState(0)
    rows = []
    t = dtcwt.Transform1d(biort=BIORT, qshift=QSHIFT)

    print(f"numpy {np.__version__} | pywt {pywt.__version__} | dtcwt {dtcwt.__version__}")
    print(f"DTCWT biort={BIORT} qshift={QSHIFT} | DWT wavelet={DWT_WAVELET}\n")

    for n in N_LIST:
        x = rng.poisson(20, size=n).astype(np.float64)
        for j in J_LIST:
            # ---------------- DTCWT ----------------
            try:
                p = t.forward(x, nlevels=j)
                low = np.asarray(p.lowpass)
                highs = [np.asarray(h) for h in p.highpasses]
                # perfect-reconstruction check (sanity)
                rec = np.asarray(t.inverse(p)).ravel()
                pr_err = float(np.max(np.abs(rec - x)))
                row = dict(transform='DTCWT', N=n, J=j, status='ok',
                           lowpass=fmt(low),
                           lowpass_len=int(low.size),
                           lowpass_is_complex=bool(np.iscomplexobj(low)),
                           highpass=' | '.join(fmt(h) for h in highs),
                           highpass_lens=','.join(str(h.size) for h in highs),
                           highpass_is_complex=all(np.iscomplexobj(h) for h in highs),
                           needs_ravel=(low.ndim == 2),
                           recon_max_abs_err=pr_err,
                           effective_level=j)
            except Exception as e:
                row = dict(transform='DTCWT', N=n, J=j, status=f'error: {e}')
            rows.append(row)

            # ---------------- DWT ------------------
            eff = dwt_safe_level(n, j)
            try:
                c = pywt.wavedec(x, DWT_WAVELET, level=j, mode='symmetric')
                row = dict(transform='DWT(db4, requested J)', N=n, J=j, status='ok',
                           lowpass=fmt(c[0]), lowpass_len=int(c[0].size),
                           lowpass_is_complex=False,
                           highpass=' | '.join(fmt(h) for h in reversed(c[1:])),
                           highpass_lens=','.join(str(h.size) for h in reversed(c[1:])),
                           highpass_is_complex=False, needs_ravel=False,
                           recon_max_abs_err=float('nan'),
                           effective_level=eff)
            except Exception as e:
                row = dict(transform='DWT(db4, requested J)', N=n, J=j,
                           status=f'error: {e}', effective_level=eff)
            rows.append(row)

    # ---------------- print --------------------
    for r in rows:
        if r['status'] != 'ok':
            print(f"{r['transform']:<22} N={r['N']:<4} J={r['J']}  {r['status']}")
            continue
        print(f"{r['transform']:<22} N={r['N']:<4} J={r['J']}  "
              f"low={r['lowpass']:<22} high(L1..LJ)={r['highpass_lens']:<14} "
              f"complex_high={r['highpass_is_complex']}"
              + (f"  DWT_safe_level(DWTAssessment)={r['effective_level']}"
                 if r['transform'].startswith('DWT') else
                 f"  recon_err={r['recon_max_abs_err']:.1e}"))

    # --- DWT effective level for the window lengths that actually occur ----
    print("\nDWTAssessment._safe_level(n, 3) for n = 32..64:")
    eff_rows = [(n, dwt_safe_level(n, 3)) for n in range(32, 65)]
    print('  ' + ', '.join(f'{n}->{l}' for n, l in eff_rows))

    if args.out:
        import csv
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        order = ['transform', 'N', 'J', 'status', 'lowpass', 'lowpass_len',
                 'lowpass_is_complex', 'highpass', 'highpass_lens',
                 'highpass_is_complex', 'needs_ravel', 'effective_level',
                 'recon_max_abs_err']
        keys = order + [k for k in keys if k not in order]
        with open(out / 'dtcwt_shapes.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(out / 'dwt_safe_level.csv', 'w', newline='', encoding='utf-8') as f:
            f.write('n,requested_level,effective_level\n')
            for n, l in eff_rows:
                f.write(f'{n},3,{l}\n')
        with open(out / 'versions.txt', 'w', encoding='utf-8') as f:
            f.write(f"numpy {np.__version__}\npywt {pywt.__version__}\n"
                    f"dtcwt {dtcwt.__version__}\npython {sys.version}\n")
        print(f"\nSaved: {out / 'dtcwt_shapes.csv'}")


if __name__ == '__main__':
    main()
