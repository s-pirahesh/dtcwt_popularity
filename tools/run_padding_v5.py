r"""
Padding strategies under protocol V5 (revision-srep-v5, task T3.4 / E8)
======================================================================
Answers R4.11 (boundary padding: explain the strategy, compare alternatives or
justify the choice).  No existing module is changed.

Two kinds of boundary handling act on a wavelet-based score (decided with
Sajjad, 26 Sep 2026, chat 11):

  pad  - explicit LEFT padding to N = 64 (protocol V5: reflect, on the oldest
         side).  It is active only while a series is shorter than 64, i.e. in
         windows 32..63.  Five modes are compared on windows 32..64 only (the
         later windows cannot change; window 64 is kept because its RSI uses
         the Top-K of window 63).  The transform extension stays 'symmetric'.
           reflect (default) | symmetric | edge | zero | periodic
         (numpy.pad modes reflect, symmetric, edge, constant 0, wrap).
         "No padding" needs no run: the default run restricted to windows >= 64.

  ext  - the boundary extension INSIDE the transform, at both ends and at every
         level, in every window (it also acts on the newest samples).
           DTCWT (WSPI, DTCWT+AF), dtcwt 0.14 numpy backend:
             symmetric (default: library, half-sample symmetric, the end
                        sample is repeated) | reflect (whole-sample symmetric,
                        no repeat) | edge (constant) | zero | periodic
             colfilter / coldfilt are re-implemented here with one index map
             per mode; 'symmetric' is the library map and gives the library
             output bit for bit (tools/test_padding_v5.py).
           DWT (DWT+AF): pywt modes symmetric (default) | reflect | constant
             (= edge) | zero | periodic.
         All windows >= 32.  Left padding stays reflect (default).

Methods: WSPI, DTCWT+AF, DWT+AF (the three wavelet-based methods share one
padding policy in protocol V5).  N = 64, J = 3, alpha = beta = 1, DTCWT
near_sym_a / qshift_a, DWT db4 with detail weight 0.1.  Exact 64-slot window,
zero fill, entry rule 32 observed rows, horizon 1 slot, RSI by item id, stable
tie-breaking, seed 42, robustness test on.  Use --causal-universe (paper).

Layout
  <out>/ext/<method>/protocol/<mode>_protocol.csv   windows >= 32
  <out>/ext/<method>/comparison/summary_{all,common}_windows.csv
  <out>/pad/<method>/protocol/<mode>_protocol.csv   windows 32..64
  <out>/pad/<method>/comparison/summary_all_windows.csv
  <out>/metadata/padding_run.json
  --collect <root>:
  <root>/padding_summary.csv   scenario x layer x method x mode x subset:
                               mean and SD of every metric.  subset
                                 all        ext: windows >= 32;
                                            pad: the pad run on 32..64 + the
                                            default ext run from 65 on
                                 pad_windows  windows 32..64 (pad layer)
                                 from_64    default run, windows >= 64
                                            (mode 'none' = no padding)
  <root>/padding_control.csv   ext/symmetric and pad/reflect against the T1.5
                               causal run of each method, window by window
  <root>/padded_share.csv      share of padded windows among the windows
                               common to the 9 methods of the T1.5 causal run
  <root>/boundary_weight.csv   per DTCWT lowpass coefficient (N=64, J=3):
                               share of its filter weight on extended samples
                               (symmetric mode) and its mu_L weight

Examples (from the project root, Windows):

  python tools\run_padding_v5.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.4_padding\youtube_hourly
  python tools\run_padding_v5.py --collect results\revision_v5\T3.4_padding

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.4.md
Unit test: tools/test_padding_v5.py
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dtcwt  # noqa: E402
import pywt  # noqa: E402
from dtcwt.coeffs import biort as _biort, qshift as _qshift  # noqa: E402
from dtcwt.numpy.lowlevel import _column_convolve  # noqa: E402
from dtcwt.utils import reflect as _reflect  # noqa: E402

from evaluation.fast_evaluator import FastMethod, _waf_rows  # noqa: E402
from evaluation.protocol_v5 import METRIC_COLUMNS, ProtocolV5Evaluator, summarize  # noqa: E402

WINDOW, LEVEL = 64, 3
FIRST, LAST_PADDED = 32, 63          # padded windows (protocol V5)
PAD_LAST_RUN = 64                    # the pad layer is run on windows 32..64
BIORT, QSHIFT = 'near_sym_a', 'qshift_a'
DWT_WAVELET, DETAIL_WEIGHT = 'db4', 0.1
METHODS = ('WSPI', 'DTCWT+AF', 'DWT+AF')
MODES = ('symmetric', 'reflect', 'edge', 'zero', 'periodic')
PAD_MODES = ('reflect', 'symmetric', 'edge', 'zero', 'periodic')
DEFAULT_EXT, DEFAULT_PAD = 'symmetric', 'reflect'
NP_PAD = {'reflect': dict(mode='reflect'), 'symmetric': dict(mode='symmetric'),
          'edge': dict(mode='edge'), 'zero': dict(mode='constant', constant_values=0.0),
          'periodic': dict(mode='wrap')}
PYWT_MODE = {'symmetric': 'symmetric', 'reflect': 'reflect', 'edge': 'constant',
             'zero': 'zero', 'periodic': 'periodic'}
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# =============================================================================
# DTCWT 1-D forward with a selectable boundary extension
# (same operations as dtcwt 0.14 numpy Transform1d.forward / colfilter / coldfilt)
# =============================================================================
def ext_index(idx, r: int, mode: str) -> np.ndarray:
    """Row index of the extended signal.  Index r points to an extra zero row."""
    idx = np.asarray(idx, dtype=np.int64)
    if mode == 'symmetric':
        return _reflect(idx, -0.5, r - 0.5)          # the library map
    if mode == 'reflect':
        return _reflect(idx, 0, r - 1)
    if mode == 'edge':
        return np.clip(idx, 0, r - 1)
    if mode == 'periodic':
        return np.mod(idx, r)
    if mode == 'zero':
        return np.where((idx < 0) | (idx >= r), r, idx)
    raise ValueError(mode)


def _with_zero_row(X):
    return np.vstack([X, np.zeros((1, X.shape[1]), dtype=X.dtype)])


def colfilter_ext(X, h, mode):
    r = X.shape[0]
    h = np.asarray(h).reshape(-1, 1)
    m2 = np.fix(h.shape[0] * 0.5)
    xe = ext_index(np.arange(-m2, r + m2, dtype=np.int64), r, mode)
    return _column_convolve(_with_zero_row(X)[xe, :], h)


def coldfilt_ext(X, ha, hb, mode):
    r, c = X.shape
    if r % 4 != 0:
        raise ValueError('No. of rows in X must be a multiple of 4')
    ha = np.asarray(ha, dtype=np.float64)
    hb = np.asarray(hb, dtype=np.float64)
    m = ha.shape[0]
    xe = ext_index(np.arange(-m, r + m), r, mode)
    hao, hae = ha[0:m:2].reshape(-1, 1), ha[1:m:2].reshape(-1, 1)
    hbo, hbe = hb[0:m:2].reshape(-1, 1), hb[1:m:2].reshape(-1, 1)
    t = np.arange(5, r + 2 * m - 2, 4)
    r2 = r // 2
    Y = np.zeros((r2, c), dtype=X.dtype)
    if np.sum(ha * hb) > 0:
        s1, s2 = slice(0, r2, 2), slice(1, r2, 2)
    else:
        s2, s1 = slice(0, r2, 2), slice(1, r2, 2)
    Xz = _with_zero_row(X)
    Y[s1, :] = _column_convolve(Xz[xe[t - 1], :], hao) + _column_convolve(Xz[xe[t - 3], :], hae)
    Y[s2, :] = _column_convolve(Xz[xe[t], :], hbo) + _column_convolve(Xz[xe[t - 2], :], hbe)
    return Y


def dtcwt_forward_ext(Xcols, nlevels: int, mode: str):
    """Columns are signals (as in dtcwt).  Returns (lowpass, [highpasses])."""
    X = np.asarray(Xcols, dtype=np.float64)
    if X.shape[0] % 2 != 0:
        raise ValueError('Size of input X must be a multiple of 2')
    h0o, g0o, h1o, g1o = _biort(BIORT)
    h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = _qshift(QSHIFT)
    Hi = colfilter_ext(X, h1o, mode)
    Lo = colfilter_ext(X, h0o, mode)
    Yh = [Hi[::2, :] + 1j * Hi[1::2, :]]
    for _ in range(1, nlevels):
        if Lo.shape[0] % 4 != 0:
            Lo = np.vstack((Lo[0, :], Lo, Lo[-1, :]))
        Hi = coldfilt_ext(Lo, h1b, h1a, mode)
        Lo = coldfilt_ext(Lo, h0b, h0a, mode)
        Yh.append(Hi[::2, :] + 1j * Hi[1::2, :])
    return Lo, Yh


# =============================================================================
# Scorers
# =============================================================================
def pad_left(X: np.ndarray, mode: str, target: int = WINDOW) -> np.ndarray:
    L = X.shape[1]
    if L >= target:
        return X
    return np.pad(X, ((0, 0), (target - L, 0)), **NP_PAD[mode])


def wspi_from_bands(low, highs, n):
    """Same operations as fast_evaluator.make_batch_wspi.core (alpha = beta = 1)."""
    lm = np.abs(low)
    k = lm.shape[1]
    w = 2.0 ** -np.arange(k)[::-1]
    mu = (lm * w).sum(axis=1) / w.sum()
    e_low = (lm ** 2).sum(axis=1)
    e_h = np.stack([(np.abs(h) ** 2).sum(axis=1) for h in highs], axis=1)
    e_tot = e_low + e_h.sum(axis=1)
    R = np.where(e_tot > 0, e_low / np.where(e_tot > 0, e_tot, 1.0), 0.0)
    E = np.concatenate([e_low[:, None], e_h], axis=1)
    tot = E.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = E / np.where(tot > 0, tot, 1.0)[:, None]
        term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
    ent = -term.sum(axis=1)
    max_ent = np.log2(E.shape[1])
    WE = np.where(tot > 0, ent / max_ent if max_ent > 0 else 0.0, 0.0)
    expo = np.zeros(n)
    expo = expo + 1.0 * R
    expo = expo - 1.0 * WE
    return mu * np.exp(expo)


def make_scorer(method: str, pad: str = DEFAULT_PAD, ext: str = DEFAULT_EXT, level: int = LEVEL):
    if method not in METHODS or pad not in PAD_MODES or ext not in MODES:
        raise ValueError((method, pad, ext))

    def f(X):
        Xp = pad_left(X, pad, max(WINDOW, 2 ** (level + 1)))
        if method == 'DWT+AF':
            c = pywt.wavedec(Xp, DWT_WAVELET, level=level, mode=PYWT_MODE[ext], axis=-1)
            return _waf_rows(c[0]) + DETAIL_WEIGHT * _waf_rows(c[-1])
        lo, hs = dtcwt_forward_ext(np.ascontiguousarray(Xp.T), level, ext)
        low = np.ascontiguousarray(lo.T)
        highs = [np.ascontiguousarray(h.T) for h in hs]
        if method == 'WSPI':
            return wspi_from_bands(low, highs, X.shape[0])
        return _waf_rows(np.abs(low)) + DETAIL_WEIGHT * _waf_rows(np.abs(highs[0]))
    f.__name__ = f'v5_{method}_pad-{pad}_ext-{ext}_L{level}_W{WINDOW}'
    return f


def boundary_weight(window: int = WINDOW, level: int = LEVEL) -> pd.DataFrame:
    """Share of the filter weight of each DTCWT lowpass coefficient that falls on
    extended samples (symmetric mode): |A_sym - A_zero| / (|A_zero| + |A_sym - A_zero|),
    row-wise L1, where A maps the window to the lowpass coefficients."""
    I = np.eye(window)
    Az = dtcwt_forward_ext(I, level, 'zero')[0]
    As = dtcwt_forward_ext(I, level, 'symmetric')[0]
    inside = np.abs(Az).sum(axis=1)
    outside = np.abs(As - Az).sum(axis=1)
    n = Az.shape[0]
    w = 2.0 ** -np.arange(n)[::-1]
    return pd.DataFrame({'coefficient': np.arange(1, n + 1),
                         'extended_weight_share': outside / (inside + outside),
                         'mu_L_weight': w / w.sum()})


# =============================================================================
# Run (one scenario)
# =============================================================================
def _write(df: pd.DataFrame, f: Path):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8')
    tmp.replace(f)


def run_one(ev, method, pad, ext, last_window=None):
    fm = FastMethod(ext if last_window is None else pad, WINDOW, 32, make_scorer(method, pad, ext))
    saved = ev.num_slots
    if last_window is not None:
        ev.num_slots = min(saved, last_window)
    try:
        df = ev.run_method(fm)
    finally:
        ev.num_slots = saved
    dur = df.attrs.get('duration_s', np.nan)
    return df[df['window_id'] >= FIRST].reset_index(drop=True), dur


def run(a):
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=a.seed,
                                      robustness=True, causal_universe=a.causal_universe)
    ev.verbose = False
    out = _abs(a.out)
    methods = a.methods or list(METHODS)
    layers = ['ext', 'pad'] if a.layer == 'both' else [a.layer]
    rt = {}
    t_all = time.time()
    for layer in layers:
        for m in methods:
            base = out / layer / m
            res = {}
            modes = MODES if layer == 'ext' else PAD_MODES
            for mode in modes:
                f = base / 'protocol' / f'{mode}_protocol.csv'
                if a.resume and f.exists():
                    res[mode] = pd.read_csv(f)
                    print(f'{layer} {m:<9} {mode:<9} (kept from an earlier run)')
                    continue
                if layer == 'ext':
                    df, dur = run_one(ev, m, DEFAULT_PAD, mode)
                else:
                    df, dur = run_one(ev, m, mode, DEFAULT_EXT, last_window=PAD_LAST_RUN)
                df['method'] = mode
                _write(df, f)
                res[mode] = df
                rt[f'{layer}/{m}/{mode}'] = round(float(dur), 2)
                print(f'{layer} {m:<9} {mode:<9} {dur:8.1f} s  windows {len(df):<6} '
                      f'ndcg@10 {df["ndcg@10"].mean():.4f}  rsi@10 {df["rsi@10"].mean():.4f}  '
                      f'robust {df["robustness_distortion"].mean():7.2f}', flush=True)
            cdir = base / 'comparison'
            cdir.mkdir(parents=True, exist_ok=True)
            summarize(res, common=False).to_csv(cdir / 'summary_all_windows.csv',
                                                index=False, encoding='utf-8')
            if layer == 'ext':
                summarize(res, common=True).to_csv(cdir / 'summary_common_windows.csv',
                                                   index=False, encoding='utf-8')
    bw = boundary_weight()
    meta = dict(task='T3.4 padding strategies (E8)', protocol='v5', mode='dense',
                window=WINDOW, level=LEVEL, alpha=1.0, beta=1.0, methods=methods, layers=layers,
                ext_modes=list(MODES), pad_modes=list(PAD_MODES),
                default=dict(pad=DEFAULT_PAD, ext=DEFAULT_EXT),
                numpy_pad={k: v['mode'] for k, v in NP_PAD.items()},
                pywt_mode=PYWT_MODE,
                dtcwt=dict(biort=BIORT, qshift=QSHIFT,
                           extension='colfilter/coldfilt re-implemented with a mode-dependent '
                                     'index map; symmetric = library map'),
                dwt=dict(wavelet=DWT_WAVELET, detail_weight=DETAIL_WEIGHT),
                pad_layer_windows=[FIRST, PAD_LAST_RUN], ext_layer_first_window=FIRST,
                padded_windows=[FIRST, LAST_PADDED],
                boundary_weight=dict(per_coefficient=[round(x, 4) for x in bw['extended_weight_share']],
                                     mu_L_weighted=round(float((bw['extended_weight_share']
                                                                * bw['mu_L_weight']).sum()), 4)),
                min_obs=32, seed=a.seed, rsi_by_item=True, robustness=True,
                causal_universe=a.causal_universe,
                universe_rule=(('total count before the test slot >= %d' if a.causal_universe
                                else 'total count over the whole file >= %d') % a.min_obs),
                tie_rule='stable sort, ties by fixed item order (item ids sorted as str)',
                horizon_slots=1, slot_minutes=ev.slot.total_seconds() / 60,
                num_items=len(ev.items), num_slots=ev.num_slots + 1,
                data=str(a.data), dataset_min_obs=a.min_obs,
                runtime_seconds=rt, runtime_total_seconds=round(time.time() - t_all, 1),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, dtcwt=dtcwt.__version__,
                              pywt=pywt.__version__),
                host=platform.platform(), created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    (out / 'metadata' / 'padding_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'\nSaved to {out}  ({time.time() - t_all:.0f} s)')


# =============================================================================
# Collect
# =============================================================================
def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Window-by-window equality of a against b cut to a's window range."""
    a = a.set_index('window_id')
    lo, hi = int(a.index.min()), int(a.index.max())
    b = b[(b['window_id'] >= lo) & (b['window_id'] <= hi)].set_index('window_id')
    same = a.index.equals(b.index)
    worst = 0.0
    if same:
        for m in METRIC_COLUMNS:
            if m in a and m in b and np.issubdtype(a[m].dtype, np.number):
                dd = np.nanmax(np.abs(a[m].to_numpy(float) - b[m].to_numpy(float)))
                worst = max(worst, float(dd) if np.isfinite(dd) else 0.0)
    return dict(n_windows=len(a), n_windows_ref=len(b), same_window_ids=bool(same),
                max_abs_diff=worst, status='equal' if same and worst < 1e-9 else 'DIFFERENT')


def _row(scen, layer, m, mode, subset, d):
    r = {'scenario': scen, 'layer': layer, 'method': m, 'mode': mode, 'subset': subset,
         'is_default': mode == (DEFAULT_EXT if layer == 'ext' else DEFAULT_PAD),
         'n_windows': len(d), 'first_window': int(d['window_id'].min()),
         'last_window': int(d['window_id'].max()),
         'padded_share': float(d['padded'].mean()) if 'padded' in d else np.nan}
    for c in METRIC_COLUMNS:
        r[f'{c}_mean'] = d[c].mean()
        r[f'{c}_sd'] = d[c].std()
    return r


def padded_share(ref_t15: Path) -> pd.DataFrame:
    rows = []
    for scen in sorted(p for p in ref_t15.iterdir() if p.is_dir()):
        files = sorted((scen / 'protocol').glob('*_protocol.csv'))
        if not files:
            continue
        runs = {f.name[:-len('_protocol.csv')]: pd.read_csv(f) for f in files}
        ids = set.intersection(*[set(d['window_id']) for d in runs.values()])
        for m in METHODS:
            if m not in runs:
                continue
            d = runs[m][runs[m]['window_id'].isin(ids)]
            p = d[d['padded'].astype(bool)]
            rows.append({'scenario': scen.name, 'method': m, 'n_methods': len(runs),
                         'common_windows': len(d), 'padded_windows': len(p),
                         'padded_share': len(p) / len(d) if len(d) else np.nan,
                         'first_padded': int(p['window_id'].min()) if len(p) else None,
                         'last_padded': int(p['window_id'].max()) if len(p) else None,
                         'source': str(scen / 'protocol')})
    return pd.DataFrame(rows)


def collect(root: Path, ref_t15: Path = None):
    ref_t15 = ref_t15 or _abs(REF_T15)
    rows, ctrl = [], []
    for scen in sorted(p for p in root.iterdir() if p.is_dir()):
        n_found = 0
        for m in METHODS:
            ext = {md: pd.read_csv(f) for md in MODES
                   if (f := scen / 'ext' / m / 'protocol' / f'{md}_protocol.csv').exists()}
            pad = {md: pd.read_csv(f) for md in PAD_MODES
                   if (f := scen / 'pad' / m / 'protocol' / f'{md}_protocol.csv').exists()}
            n_found += len(ext) + len(pad)
            for md, d in ext.items():
                rows.append(_row(scen.name, 'ext', m, md, 'all', d))
            base = ext.get(DEFAULT_EXT)
            for md, d in pad.items():
                rows.append(_row(scen.name, 'pad', m, md, 'pad_windows', d))
                if base is not None:
                    comb = pd.concat([d, base[base['window_id'] > int(d['window_id'].max())]],
                                     ignore_index=True)
                    rows.append(_row(scen.name, 'pad', m, md, 'all', comb))
            if base is not None:
                rows.append(_row(scen.name, 'pad', m, 'none', 'from_64',
                                 base[base['window_id'] >= LAST_PADDED + 1]))
            ref = ref_t15 / scen.name / 'protocol' / f'{m}_protocol.csv'
            for layer, runs, md in (('ext', ext, DEFAULT_EXT), ('pad', pad, DEFAULT_PAD)):
                c = {'scenario': scen.name, 'layer': layer, 'method': m, 'mode': md,
                     'reference': 'T1.5 causal', 'reference_file': str(ref)}
                if md not in runs or not ref.exists():
                    c['status'] = 'missing'
                else:
                    c.update(compare_frames(runs[md], pd.read_csv(ref)))
                ctrl.append(c)
        if n_found:
            st = [c['status'] for c in ctrl if c['scenario'] == scen.name]
            print(f'{scen.name}: {n_found} protocol files; control {st}')
    pd.DataFrame(rows).to_csv(root / 'padding_summary.csv', index=False, encoding='utf-8')
    pd.DataFrame(ctrl).to_csv(root / 'padding_control.csv', index=False, encoding='utf-8')
    if ref_t15.exists():
        padded_share(ref_t15).to_csv(root / 'padded_share.csv', index=False, encoding='utf-8')
    boundary_weight().to_csv(root / 'boundary_weight.csv', index=False, encoding='utf-8')
    print(f'Saved padding_summary.csv, padding_control.csv, padded_share.csv and '
          f'boundary_weight.csv in {root}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int, help='dataset min_observations (catalogue filter)')
    ap.add_argument('--out', help='scenario folder')
    ap.add_argument('--layer', choices=['ext', 'pad', 'both'], default='both')
    ap.add_argument('--methods', nargs='*', choices=list(METHODS), default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--resume', action='store_true', help='keep protocol CSVs already written')
    ap.add_argument('--collect', help='T3.4 root: write the four collect tables')
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect))
    else:
        if not (a.data and a.min_obs is not None and a.out):
            ap.error('--data, --min-obs and --out are required')
        run(a)


if __name__ == '__main__':
    main()
