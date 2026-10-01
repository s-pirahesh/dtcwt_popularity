r"""
Relation between the WSPI features R and WE
===========================================
Questions: are R and WE complementary, what distinct information does
each feature give, and why are the three features enough?  No existing module
is changed; the features are computed with the exact WSPI code of protocol V5.

Design (27 Sep 2026)
  * Item-windows: exactly those that WSPI scores in the paper setting: protocol
    V5, N = 64, J = 3, eligibility 32 observed rows, causal catalogue
    (--causal-universe), windows >= 32 (reflect padding only in windows 32..63,
    as in every V5 run).  Ground truth = the count of the next slot (horizon 1).
  * Features per item-window (same operations as run_param_grid.WSPIFeatureCache,
    checked bit for bit by tools/test_feature_relation.py):
        mu    recency-weighted mean of |lowpass|         (mu_L)
        R     E_low / E_total
        WE    normalised Shannon entropy of the J+1 band energies
        D     normalised entropy of the J detail energies only,
              D = H(q) / log2 J,  q_j = E_j / sum_{m>=1} E_m   (0 if no detail energy)
    Exact identity (checked in every window, metadata identity_max_abs):
        WE * log2(J+1) = h(R) + (1 - R) * H(q),   h = binary entropy
    so WE = A + B with A = h(R) / log2(J+1) (a function of R alone) and
    B = (1 - R) H(q) / log2(J+1) (the only part of WE that R does not fix).
    Item-windows with zero energy (all-zero window) have no R, WE or D; they are
    counted (n_zero) and left out of every relation statistic, but kept in the
    NDCG@10 control.
  * Within each window (this is where ranking happens), on ranks:
        sp_R_WE, pe_R_WE        Spearman / Pearson correlation of R and WE
        sp_R_logmu, sp_WE_logmu, sp_R_D, sp_mu_y
        pc_R_y_mu               partial Spearman correlation of R with the next-slot
                                count y, controlling mu (rank residuals)
        pc_WE_y_mu, pc_S_y_mu   same for WE and for S = R - WE (the exponent, alpha = beta = 1)
        pc_R_y_muWE             R after mu and WE    (unique part of R)
        pc_WE_y_muR             WE after mu and R    (unique part of WE)
        pc_D_y_muR              D after mu and R     (the part of WE that R does not fix)
        mean_R, mean_WE, mean_D, n_valid, n_zero, ndcg@10 (control)
    Windows with fewer than 10 valid items get NaN statistics.  When the next-slot
    count is the same for every valid item, the statistics that involve y are NaN;
    a partial correlation is NaN when nothing is left after the controls (relative
    residual energy <= 1e-12).  Before this rule (fixed 27 Sep 2026) such
    a case returned rounding noise (YouTube window 277).
  * Pooled over all item-windows of the scenario (descriptive):
        quantiles of R, WE, D; share_R_gt_half; Pearson / Spearman of R and WE,
        R and log mu, WE and log mu, R and D; eta2_WE_given_R (correlation ratio,
        50 equal-count bins of R); r2_WE_A (squared Pearson of WE and A);
        share_var_B = var(B) / var(WE); mutual information on ranks (20 x 20
        equal-count bins, bits) for (R, WE) and (R, D), with the Gaussian
        equivalent -0.5 log2(1 - rho_s^2); VIF of log mu, R and WE.
        density.csv: 2-D histogram of (R, WE), 100 x 100 cells on [0, 1]^2
        (non-empty cells only), for the SI figure.

Layout
  <out>/protocol/feature_relation_protocol.csv   one row per window >= 32
  <out>/pooled_summary.csv                       one row
  <out>/density.csv
  <out>/metadata/feature_relation_run.json
  --collect <root>
  <root>/feature_relation_summary.csv   scenario x statistic: pooled value, or the
                                        mean over windows with a 95 % circular block
                                        bootstrap interval (B 10000, seed 42, paper
                                        blocks; sens = one-day blocks for taxi)
  <root>/feature_relation_control.csv   window set and NDCG@10 against the main
                                        causal WSPI run (must be equal)
  <root>/alpha_beta_total.csv           rows of the alpha x beta grid (part all) with
                                        alpha + beta in {1, 2}, and (0, 0): the
                                        effect of the split between R and WE at a
                                        fixed total weight (no new run)

Examples (from the project root, Windows):

  python tools\run_feature_relation.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.7_feature_relation\youtube_hourly
  python tools\run_feature_relation.py --collect results\revision_v5\T3.7_feature_relation

Full command list: REPRODUCE.md
Unit test: tools/test_feature_relation.py
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
sys.path.insert(0, str(ROOT / 'tools'))

import dtcwt  # noqa: E402

from evaluation.fast_evaluator import _dtcwt_forward_rows  # noqa: E402
from evaluation.protocol_v5 import (ProtocolV5Evaluator, _pad_left_reflect,  # noqa: E402
                                    _target_len, ndcg_stable, stable_order)
from stats_report import cbb_means, cbb_starts, percentile_ci  # noqa: E402

WINDOW, LEVEL, MIN_OBS, FIRST_WINDOW = 64, 3, 32, 32
BIORT, QSHIFT = 'near_sym_a', 'qshift_a'
MIN_VALID = 10
RESID_TOL = 1e-12                          # relative residual energy below which a partial corr is NaN
ETA_BINS, MI_BINS, DENS_BINS = 50, 20, 100
QUANTILES = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
B_BOOT, SEED = 10000, 42
REF_T15 = Path('results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
GRID_ROOT = Path('results/revision_v5/T3.2_param_grid')
DATASETS = {  # scenario folder -> (block, block_sens) of the paper statistics
    'youtube_hourly': (24, None),
    'taxi_hourly': (168, 24),
    'taxi_30min': (336, 48),
    'taxi_5min': (2016, 288),
}
WINDOW_STATS = ['sp_R_WE', 'pe_R_WE', 'sp_R_logmu', 'sp_WE_logmu', 'sp_R_D', 'sp_mu_y',
                'pc_R_y_mu', 'pc_WE_y_mu', 'pc_S_y_mu', 'pc_R_y_muWE', 'pc_WE_y_muR',
                'pc_D_y_muR', 'mean_R', 'mean_WE', 'mean_D']
Y_STATS = ['sp_mu_y', 'pc_R_y_mu', 'pc_WE_y_mu', 'pc_S_y_mu', 'pc_R_y_muWE', 'pc_WE_y_muR',
           'pc_D_y_muR']
PROTOCOL_COLUMNS = ['window_id', 'timestamp', 'num_items', 'n_valid', 'n_zero', 'ndcg@10'] + WINDOW_STATS


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# =============================================================================
# Features
# =============================================================================
class FeatureMaker:
    """Band energies and the WSPI features of a window matrix (protocol V5 padding).
    mu, R and WE use the operations of run_param_grid.WSPIFeatureCache._compute."""

    def __init__(self, window: int = WINDOW, level: int = LEVEL):
        self.tr = dtcwt.Transform1d(biort=BIORT, qshift=QSHIFT)
        self.level = level
        self.tgt = _target_len(window, level)

    def __call__(self, X: np.ndarray) -> dict:
        Xp = _pad_left_reflect(X, self.tgt)
        low, highs = _dtcwt_forward_rows(self.tr, Xp, self.level)
        lm = np.abs(low)
        n = lm.shape[1]
        w = 2.0 ** -np.arange(n)[::-1]
        mu = (lm * w).sum(axis=1) / w.sum()
        e_low = (lm ** 2).sum(axis=1)
        e_h = np.stack([(np.abs(h) ** 2).sum(axis=1) for h in highs], axis=1)
        e_tot = e_low + e_h.sum(axis=1)
        R = np.where(e_tot > 0, e_low / np.where(e_tot > 0, e_tot, 1.0), 0.0)
        E = np.concatenate([e_low[:, None], e_h], axis=1)
        tot = E.sum(axis=1)
        WE = np.where(tot > 0, _entropy_bits(E) / np.log2(E.shape[1]), 0.0)
        ed = e_h.sum(axis=1)
        Hq = np.where(ed > 0, _entropy_bits(e_h), 0.0)
        D = Hq / np.log2(e_h.shape[1])
        return dict(mu=mu, R=R, WE=WE, D=D, Hq=Hq, e_tot=e_tot)


def _entropy_bits(E: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of each row of non-negative energies (same
    operations as WSPIFeatureCache; 0 for an all-zero row)."""
    tot = E.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = E / np.where(tot > 0, tot, 1.0)[:, None]
        term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
    return -term.sum(axis=1)


def binary_entropy(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(R > 0, R * np.log2(np.where(R > 0, R, 1.0)), 0.0)
        b = np.where(R < 1, (1 - R) * np.log2(np.where(R < 1, 1 - R, 1.0)), 0.0)
    return -(a + b)


def identity_residual(f: dict, level: int = LEVEL) -> np.ndarray:
    """WE log2(J+1) - h(R) - (1-R) H(q); 0 up to rounding for every valid row."""
    return f['WE'] * np.log2(level + 1) - binary_entropy(f['R']) - (1 - f['R']) * f['Hq']


# =============================================================================
# Rank statistics
# =============================================================================
def ranks(x) -> np.ndarray:
    """Average ranks (ties share the mean rank), as scipy.stats.rankdata."""
    return pd.Series(np.asarray(x, dtype=np.float64)).rank(method='average').to_numpy()


def corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64) - np.mean(a)
    b = np.asarray(b, dtype=np.float64) - np.mean(b)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else float('nan')


def partial_corr(x, y, controls) -> float:
    """Correlation of x and y after removing (least squares, with intercept)
    the linear effect of the control columns from both."""
    A = np.column_stack([np.ones(len(x))] + list(controls))
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    for v, r in ((x, rx), (y, ry)):          # nothing left after the controls -> undefined
        vc = v - np.mean(v)
        if vc @ vc == 0 or r @ r <= RESID_TOL * (vc @ vc):
            return float('nan')
    return corr(rx, ry)


def window_stats(f: dict, y: np.ndarray) -> dict:
    """Relation statistics of one window (valid items only, on ranks)."""
    ok = f['e_tot'] > 0
    out = {'n_valid': int(ok.sum()), 'n_zero': int((~ok).sum())}
    if ok.sum() < MIN_VALID:
        out.update({c: float('nan') for c in WINDOW_STATS})
        return out
    R, WE, D, mu, yy = f['R'][ok], f['WE'][ok], f['D'][ok], f['mu'][ok], y[ok]
    rR, rW, rD, rM, rY = ranks(R), ranks(WE), ranks(D), ranks(mu), ranks(yy)
    rS = ranks(R - WE)
    y_const = bool(np.all(rY == rY[0]))       # same next-slot count for every item
    out.update(sp_R_WE=corr(rR, rW), pe_R_WE=corr(R, WE),
               sp_R_logmu=corr(rR, rM), sp_WE_logmu=corr(rW, rM), sp_R_D=corr(rR, rD),
               sp_mu_y=corr(rM, rY),
               pc_R_y_mu=partial_corr(rR, rY, [rM]), pc_WE_y_mu=partial_corr(rW, rY, [rM]),
               pc_S_y_mu=partial_corr(rS, rY, [rM]),
               pc_R_y_muWE=partial_corr(rR, rY, [rM, rW]),
               pc_WE_y_muR=partial_corr(rW, rY, [rM, rR]),
               pc_D_y_muR=partial_corr(rD, rY, [rM, rR]),
               mean_R=float(R.mean()), mean_WE=float(WE.mean()), mean_D=float(D.mean()))
    if y_const:                               # no ranking of the next slot: y statistics undefined
        out.update({c: float('nan') for c in Y_STATS})
    return out


def mutual_info_ranks(a, b, bins: int = MI_BINS) -> float:
    """Mutual information (bits) of the equal-count bin indices of a and b."""
    n = len(a)
    qa = np.minimum((ranks(a) - 1) * bins // n, bins - 1).astype(int)
    qb = np.minimum((ranks(b) - 1) * bins // n, bins - 1).astype(int)
    P = np.bincount(qa * bins + qb, minlength=bins * bins).reshape(bins, bins) / n
    pa, pb = P.sum(axis=1, keepdims=True), P.sum(axis=0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pa @ pb)[nz])).sum())


def eta_squared(y, x, bins: int = ETA_BINS) -> float:
    """Correlation ratio: share of var(y) explained by the mean of y in
    equal-count bins of x."""
    n = len(x)
    q = np.minimum((ranks(x) - 1) * bins // n, bins - 1).astype(int)
    y = np.asarray(y, dtype=np.float64)
    m = np.bincount(q, weights=y, minlength=bins) / np.maximum(np.bincount(q, minlength=bins), 1)
    return float(1.0 - np.var(y - m[q]) / np.var(y))


def vif(Z: np.ndarray, names) -> dict:
    out = {}
    for j, nm in enumerate(names):
        o = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(o)), o])
        res = Z[:, j] - A @ np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
        out[f'vif_{nm}'] = float(np.var(Z[:, j]) / np.var(res))
    return out


def pooled_stats(R, WE, D, lmu, level: int = LEVEL) -> dict:
    R, WE, D, lmu = (np.asarray(v, dtype=np.float64) for v in (R, WE, D, lmu))
    out = {'n_item_windows': int(len(R)), 'share_R_gt_half': float((R > 0.5).mean())}
    for nm, v in (('R', R), ('WE', WE), ('D', D)):
        for q, val in zip(QUANTILES, np.quantile(v, QUANTILES)):
            out[f'{nm}_q{int(round(q * 100)):02d}'] = float(val)
        out[f'{nm}_mean'] = float(v.mean())
    rR, rW, rM, rD = ranks(R), ranks(WE), ranks(lmu), ranks(D)
    out.update(pe_R_WE=corr(R, WE), sp_R_WE=corr(rR, rW),
               sp_R_logmu=corr(rR, rM), sp_WE_logmu=corr(rW, rM), sp_R_D=corr(rR, rD))
    A = binary_entropy(R) / np.log2(level + 1)
    Bp = WE - A
    out.update(eta2_WE_given_R=eta_squared(WE, R),
               r2_WE_A=corr(WE, A) ** 2,
               share_var_B=float(np.var(Bp) / np.var(WE)),
               mi_R_WE_bits=mutual_info_ranks(R, WE),
               mi_R_WE_gauss_eq_bits=float(-0.5 * np.log2(1 - out['sp_R_WE'] ** 2)),
               mi_R_D_bits=mutual_info_ranks(R, D))
    out.update(vif(np.column_stack([lmu, R, WE]), ['logmu', 'R', 'WE']))
    return out


def density(R, WE, bins: int = DENS_BINS) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    H = np.histogram2d(np.asarray(R, np.float64), np.asarray(WE, np.float64),
                       bins=[edges, edges])[0]
    i, j = np.nonzero(H)
    return pd.DataFrame({'r_lo': edges[i], 'r_hi': edges[i + 1], 'we_lo': edges[j],
                         'we_hi': edges[j + 1], 'count': H[i, j].astype(np.int64)})


# =============================================================================
# One scenario
# =============================================================================
def run(a):
    t0 = time.time()
    ev = ProtocolV5Evaluator.from_csv(_abs(a.data), dataset_min_obs=a.min_obs, seed=SEED,
                                      robustness=False, causal_universe=a.causal_universe)
    ev.verbose = False
    out = _abs(a.out)
    fm = FeatureMaker()
    recs, store = [], {k: [] for k in ('R', 'WE', 'D', 'lmu')}
    id_max, n_zero_total = 0.0, 0
    cum = np.zeros(len(ev.items)) if ev.causal_universe else None
    for k in range(ev.num_slots + 1):                # same loop as ProtocolV5Evaluator.run_method
        if cum is not None and k > 0:
            cum += ev.V[:, k - 1]
        lo, hi = ev.window_bounds(k, WINDOW)
        if ev.any_cum[hi] - ev.any_cum[lo] == 0 or not ev.any_row[k]:
            continue
        observed = ev.C[:, hi] - ev.C[:, lo]
        ok = observed >= MIN_OBS
        if cum is not None:
            ok &= cum >= ev.universe_min_count
        idx = np.where(ok)[0]
        if len(idx) < 2 or k < FIRST_WINDOW:
            continue
        X = ev.V[idx, lo:hi]
        f = fm(X)
        y = ev.V[idx, k]
        score = f['mu'] * np.exp(f['R'] - f['WE'])     # WSPI, alpha = beta = 1
        rec = {'window_id': k, 'timestamp': int((ev.start + k * ev.slot).timestamp() * 1000),
               'num_items': len(idx), 'ndcg@10': ndcg_stable(stable_order(score), y, 10)}
        rec.update(window_stats(f, y))
        recs.append(rec)
        v = f['e_tot'] > 0
        n_zero_total += int((~v).sum())
        if v.any():
            id_max = max(id_max, float(np.max(np.abs(identity_residual(f)[v]))))
            store['R'].append(f['R'][v].astype(np.float32))
            store['WE'].append(f['WE'][v].astype(np.float32))
            store['D'].append(f['D'][v].astype(np.float32))
            store['lmu'].append(np.log(np.maximum(f['mu'][v], 1e-12)).astype(np.float32))
    if not recs:
        sys.exit('no window evaluated')
    (out / 'protocol').mkdir(parents=True, exist_ok=True)
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs, columns=PROTOCOL_COLUMNS).to_csv(
        out / 'protocol' / 'feature_relation_protocol.csv', index=False, encoding='utf-8')
    arr = {k: np.concatenate(v) for k, v in store.items()}
    ps = pooled_stats(arr['R'], arr['WE'], arr['D'], arr['lmu'])
    ps['n_zero_energy'] = n_zero_total
    pd.DataFrame([ps]).to_csv(out / 'pooled_summary.csv', index=False, encoding='utf-8')
    density(arr['R'], arr['WE']).to_csv(out / 'density.csv', index=False, encoding='utf-8')
    meta = dict(task='T3.7 relation between R and WE (E7)',
                window=WINDOW, level=LEVEL, min_obs=MIN_OBS, first_window=FIRST_WINDOW,
                dtcwt=dict(biort=BIORT, qshift=QSHIFT, padding='reflect on the left to 64 '
                           '(windows 32..63 only), library symmetric extension'),
                ground_truth='count of the next slot (horizon 1)',
                features_code='same operations as tools/run_param_grid.WSPIFeatureCache._compute',
                identity='WE*log2(J+1) = h(R) + (1-R)*H(q)', identity_max_abs=id_max,
                min_valid_items=MIN_VALID, eta_bins=ETA_BINS, mi_bins=MI_BINS,
                density_bins=DENS_BINS, pooled_storage='float32',
                n_windows=len(recs), n_item_windows_valid=int(len(arr['R'])),
                n_zero_energy=n_zero_total,
                causal_universe=a.causal_universe,
                universe_rule=(('total count before the test slot >= %d' if a.causal_universe
                                else 'total count over the whole file >= %d') % a.min_obs),
                slot_minutes=ev.slot.total_seconds() / 60, num_items=len(ev.items),
                data=str(a.data), dataset_min_obs=a.min_obs,
                runtime_seconds=round(time.time() - t0, 1),
                versions=_versions(), host=platform.platform(),
                created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'metadata' / 'feature_relation_run.json').write_text(json.dumps(meta, indent=2),
                                                                encoding='utf-8')
    d = pd.DataFrame(recs)
    print(f'{out.name}: {len(d)} windows, {len(arr["R"])} valid item-windows '
          f'({n_zero_total} zero), {meta["runtime_seconds"]} s, identity max {id_max:.2e}')
    print(f'  pooled: sp_R_WE {ps["sp_R_WE"]:.4f}  eta2_WE|R {ps["eta2_WE_given_R"]:.4f}  '
          f'VIF R {ps["vif_R"]:.2f} WE {ps["vif_WE"]:.2f}')
    print('  within-window means: ' + '  '.join(f'{c} {d[c].mean():.4f}' for c in
                                                ('sp_R_WE', 'pc_R_y_mu', 'pc_WE_y_mu', 'pc_S_y_mu')))


def _versions():
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                dtcwt=dtcwt.__version__, scipy=scipy.__version__)


# =============================================================================
# Collect
# =============================================================================
def scenario_dirs(root: Path):
    return [(s, root / s) for s in DATASETS if (root / s / 'pooled_summary.csv').exists()]


def boot_ci(x: np.ndarray, block: int):
    starts = cbb_starts(len(x), block, B_BOOT, SEED)
    return percentile_ci(cbb_means(x, block, starts))


def collect(root: Path, ref_t15: Path, grid_root: Path):
    rows, ctrl = [], []
    for sc, d in scenario_dirs(root):
        block, block_sens = DATASETS[sc]
        ps = pd.read_csv(d / 'pooled_summary.csv').iloc[0]
        for stat, val in ps.items():
            rows.append(dict(scenario=sc, kind='pooled', statistic=stat, value=float(val)))
        p = pd.read_csv(d / 'protocol' / 'feature_relation_protocol.csv').sort_values('window_id')
        for stat in WINDOW_STATS:
            x = p[stat].to_numpy(dtype=np.float64)
            n_nan = int(np.isnan(x).sum())
            x = x[~np.isnan(x)]
            r = dict(scenario=sc, kind='within_window', statistic=stat, value=float(x.mean()),
                     sd=float(x.std(ddof=1)), n_windows=len(x), n_nan=n_nan, block=block)
            r['ci_low'], r['ci_high'] = boot_ci(x, block)
            if block_sens:
                r['block_sens'] = block_sens
                r['ci_low_sens'], r['ci_high_sens'] = boot_ci(x, block_sens)
            r['ci_excludes_0'] = bool(r['ci_low'] > 0 or r['ci_high'] < 0)
            rows.append(r)
        ref_f = _abs(ref_t15) / sc / 'protocol' / 'WSPI_protocol.csv'
        c = dict(scenario=sc, n_windows=len(p))
        if ref_f.exists():
            ref = pd.read_csv(ref_f)
            m = p.merge(ref, on='window_id', suffixes=('', '_ref'))
            c.update(n_windows_ref=len(ref), same_window_set=set(p['window_id']) == set(ref['window_id']),
                     max_abs_diff_ndcg10=float((m['ndcg@10'] - m['ndcg@10_ref']).abs().max()),
                     max_abs_diff_num_items=int((m['num_items'] - m['num_items_ref']).abs().max()))
            c['pass'] = bool(c['same_window_set'] and c['max_abs_diff_ndcg10'] <= 1e-12
                             and c['max_abs_diff_num_items'] == 0)
        else:
            c.update(note=f'reference missing: {ref_f}', **{'pass': False})
        ctrl.append(c)
    if not rows:
        sys.exit(f'no scenario folder with pooled_summary.csv under {root}')
    pd.DataFrame(rows).to_csv(root / 'feature_relation_summary.csv', index=False, encoding='utf-8')
    cd = pd.DataFrame(ctrl)
    cd.to_csv(root / 'feature_relation_control.csv', index=False, encoding='utf-8')
    print(cd.to_string(index=False))
    gf = _abs(grid_root) / 'grid_summary.csv'
    if gf.exists():
        g = pd.read_csv(gf)
        g = g[g['part'] == 'all'].copy()
        g['total'] = (g['alpha'] + g['beta']).round(4)
        keep = g['total'].isin([1.0, 2.0]) | ((g['alpha'] == 0) & (g['beta'] == 0))
        g = g[keep]
        g['share_beta'] = np.where(g['total'] > 0, g['beta'] / g['total'].where(g['total'] > 0, 1), np.nan)
        cols = ['scenario', 'alpha', 'beta', 'total', 'share_beta', 'n_windows',
                'ndcg@10_mean', 'spearman_rho_mean', 'rsi@10_mean', 'robustness_distortion_mean']
        g[cols].sort_values(['scenario', 'total', 'alpha']).to_csv(
            root / 'alpha_beta_total.csv', index=False, encoding='utf-8')
        print(f'alpha_beta_total.csv written from {gf}')
    else:
        print(f'[warn] T3.2 grid summary missing: {gf}; alpha_beta_total.csv not written')
    s = pd.DataFrame(rows)
    show = s[s['statistic'].isin(['sp_R_WE', 'eta2_WE_given_R', 'vif_R', 'vif_WE', 'pc_R_y_mu',
                                  'pc_WE_y_mu', 'pc_S_y_mu', 'pc_D_y_muR'])]
    print(show.pivot_table(index='statistic', columns='scenario', values='value').round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data', help='dataset CSV (one scenario)')
    ap.add_argument('--min-obs', type=int, help='catalogue threshold (YouTube 50, taxi 24)')
    ap.add_argument('--causal-universe', action='store_true', help='paper setting (T1.5)')
    ap.add_argument('--out', help='output folder of the scenario')
    ap.add_argument('--collect', help='T3.7 root folder: summary, control, alpha_beta_total')
    ap.add_argument('--ref-t15', default=str(REF_T15))
    ap.add_argument('--grid-root', default=str(GRID_ROOT))
    a = ap.parse_args()
    if a.collect:
        collect(_abs(a.collect), Path(a.ref_t15), Path(a.grid_root))
        return
    if not (a.data and a.min_obs and a.out):
        ap.error('--data, --min-obs and --out are required (or --collect)')
    if not a.causal_universe:
        print('[warn] --causal-universe is off; the paper uses it')
    run(a)


if __name__ == '__main__':
    main()
