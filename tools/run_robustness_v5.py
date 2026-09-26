r"""
Extended robustness study under protocol V5 (revision-srep-v5, task T3.5 / E4)
=============================================================================
Answers R1.4 (spike size, position and duration) and part of R4.15 (uniform,
repeatable configuration).  No existing module is changed; the scorers come
from evaluation/protocol_v5.py and the statistics from tools/stats_report.py
(imported, unchanged).

Design (decided in chat 12, 26 Sep 2026; tracker section E)
------------------------------------------------------------
"Common perturbation": in every evaluation window every method sees the SAME
corrupted data.  This extends the T1.4 rule "fixed seed and identical input
in the robustness test".

  window k      k >= 32, a row exists at slot k, and every method of both
                configurations has >= 2 eligible items (protocol V5 rules:
                exact W-slot slice, zero fill, min_obs 3 for the baselines and
                32 for the wavelet-based methods, causal item catalogue).
  pool          items eligible for ALL methods of both configurations
                (baselines W=7 and W=64, wavelet-based W=64).
  reference     the last 64 slots [k-64, k) (fewer while k < 64).
  candidates    pool items with mean_64 > 0 and mean_64 <= median of the
                non-zero mean_64 values (the "low-activity" rule of the V4 /
                V5 test; the "low variance" condition of the V4 docstring was
                never implemented and is not used).
  targets       min(50, #candidates) candidates drawn without replacement with
                numpy default_rng([seed, k]).  Same targets for every method.
  spike         magnitude per slot = max(size * mean_64, 1), added to the
                target's own series.  Position is the AGE of the newest spiked
                slot counted back from the test slot: last = 1, middle = 32,
                first = 64, random = uniform {1..L} per target
                (default_rng([seed, k, 1])).  A spike of duration d covers
                ages a, a-1, ..., a-d+1 with a = max(age, d), capped at L.
                A method with window W sees only the last W slots, so a spike
                older than W has no effect on it (this is part of the answer).
  noise         continuous noise on the whole 64-slot matrix of ALL items,
                one realisation per (seed, k), shared by every method:
                  poisson       x' ~ Poisson(x)              (default_rng([seed,k,2]))
                  gauss_snrS    x' = max(0, x + N(0, s_i^2)),
                                s_i^2 = mean_64(x_i^2) / 10^(S/10), S in {10, 0} dB
                                                             (default_rng([seed,k,3|4]))
                Eligibility and the catalogue are taken from the clean data.
  grid          one factor at a time around the V4 test (10x, last, 1 slot):
                size {2,5,10,20,50}x (last, 1 slot); position middle / first /
                random (10x, 1 slot); duration 3 / 6 slots (10x, last); plus
                the three noise settings.  13 conditions.
  seeds         42..46.  Every window value is the mean over the 5 seeds; the
                per-seed means over the windows are written separately.
  stride        --stride s evaluates only windows with k % s == 0 (taxi 5-min:
                12, i.e. one window per hour).  A window's values do not
                depend on the stride.  Statistics then use block lengths
                divided by s (block = one week / one day of windows).
  one pass      all requested methods are computed in one pass over the
                windows; targets, ages, spikes and noise are built once per
                (window, seed).  --methods splits the work over processes
                with identical results.

Metrics per window (mean over targets, then over seeds)
  dr_<c>     mean |rank_noisy - rank_clean| of the targets.  Rank = position
             in the method's own eligible set, stable order of protocol V5.
             For a spike only the target's score changes (every scorer is
             row-wise); for noise every item is re-scored.
  sdr_<c>    mean signed change (negative = the target moved UP).
  up_<c>     share of targets that moved up.
  top10_<c>  noise only: Jaccard of the clean and noisy Top-10 sets.
  WSPI only: dlogmu_<c> and dexpo_<c>, mean change of log mu_L and of the
             exponent alpha*R - beta*WE of the targets (log WSPI = log mu_L +
             exponent), to explain where the displacement comes from.  Targets
             whose noisy series is all zeros (log 0) are left out of these two
             means only.
  Control columns: num_items, npool, ntargets, ndcg@10 of the clean scores
  (must equal the T1.5 run for the default configuration and the T2.2 W064 run
  for the baselines at 64).

Configurations
  default  AF, EWMA, RRD, VSE, CompoundPop, PFRF with W=7; DWT+AF, DTCWT+AF,
           WSPI with W=64 (paper table 1)
  eq64     all nine methods with W=64 (paper table 2).  The wavelet-based
           methods are identical in both configurations (same pool, targets
           and noise), so they are computed once.

Layout
  <out>/robustness/<method>_W<NNN>.csv       one row per window (seed means, and
                                             dr_<c>_s<seed> per seed)
  <out>/robustness/<method>_W<NNN>.partial.csv   while running (checkpoint every
                                             --checkpoint windows; --resume goes on
                                             from the last written window)
  <out>/metadata/robustness_run.json
  --collect <root>:
  <root>/robustness_summary.csv   scenario x config x method x condition
  <root>/robustness_control.csv   clean NDCG@10 and num_items against the
                                  reference runs, window by window (must be equal)
  <root>/robustness_main_check.csv  default configuration: mean Delta Rank of the
                                  V4/V5 test of the paper tables (T1.5 runs) and of
                                  the common test at 10x / last / 1 slot, on the
                                  same windows, with the rank of each method
  --audit <scenario> --data ... --min-obs ... --causal-universe:
  <audit-out>/current_test_audit.csv  the V4/V5 test of the paper tables replayed
                                  (per-method targets and spike): target Jaccard
                                  between methods, median spike, zero share, and
                                  control against the T1.5 robustness column
  --stats <root> --stats-out <dir>:
  <dir>/<scenario>/<config>/<condition>/{method_summary,paired_tests}.csv
  (stats_report.analyse, reference WSPI, metrics robustness_distortion and,
  for noise, top10_overlap; Holm family = scenario x config x condition x metric)

Examples (from the project root, Windows):

  python tools\run_robustness_v5.py --data data\datasets\youtube_hourly.csv --min-obs 50 ^
         --causal-universe --out results\revision_v5\T3.5_robustness\youtube_hourly
  python tools\run_robustness_v5.py --collect results\revision_v5\T3.5_robustness
  python tools\run_robustness_v5.py --stats results\revision_v5\T3.5_robustness ^
         --stats-out results\revision_v5\T1.6_stats\T3.5_robustness

Full command list: Revisions/V4/Response/Runbooks/RUN_T3.5.md
Unit test: tools/test_robustness_v5.py
"""
import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v5 import (ProtocolV5Evaluator, build_v5_methods,  # noqa: E402
                                    make_v5_wspi, stable_order, ndcg_stable)

REF = 64
SEEDS = (42, 43, 44, 45, 46)
N_TARGETS = 50
BASELINES = ('AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF')
WAVELETS = ('DWT+AF', 'DTCWT+AF', 'WSPI')
ORDER = BASELINES + WAVELETS
POS_AGE = {'last': 1, 'middle': 32, 'first': 64}

# (name, size, position, duration)
SPIKES = [('size2', 2, 'last', 1), ('size5', 5, 'last', 1), ('size10', 10, 'last', 1),
          ('size20', 20, 'last', 1), ('size50', 50, 'last', 1),
          ('pos_middle', 10, 'middle', 1), ('pos_first', 10, 'first', 1),
          ('pos_random', 10, 'random', 1),
          ('dur3', 10, 'last', 3), ('dur6', 10, 'last', 6)]
NOISES = [('noise_poisson', 'poisson', None), ('noise_gauss_snr10', 'gauss', 10),
          ('noise_gauss_snr0', 'gauss', 0)]
CONDITIONS = [s[0] for s in SPIKES] + [n[0] for n in NOISES]

DATASETS = {  # scenario folder -> (data file, dataset min_obs, block, block_sens)
    'youtube_hourly': ('youtube_hourly.csv', 50, 24, None),
    'taxi_hourly': ('yellow_taxi_2025_all_hourly.csv', 24, 168, 24),
    'taxi_30min': ('yellow_taxi_2025_all_30min.csv', 24, 336, 48),
    'taxi_5min': ('yellow_taxi_2025_all_5min.csv', 24, 2016, 288),
}


# =============================================================================
# Building blocks (tested in tools/test_robustness_v5.py)
# =============================================================================
def all_methods():
    """{(name, W): FastMethod} for both configurations (wavelet-based once)."""
    d = build_v5_methods()
    e = build_v5_methods(window_override={n: 64 for n in BASELINES}, names=list(BASELINES))
    out = {(n, m.window_slots): m for n, m in d.items()}
    out.update({(n, 64): m for n, m in e.items()})
    return out


def config_members(config: str):
    if config == 'default':
        return [(n, 7) for n in BASELINES] + [(n, 64) for n in WAVELETS]
    if config == 'eq64':
        return [(n, 64) for n in ORDER]
    raise ValueError(config)


def spike_matrix(L: int, mags: np.ndarray, position: str, duration: int,
                 ages_random: np.ndarray) -> np.ndarray:
    """Additive perturbation (n_targets x L), oldest -> newest column."""
    n = len(mags)
    D = np.zeros((n, L))
    a = (np.asarray(ages_random, dtype=np.int64) if position == 'random'
         else np.full(n, POS_AGE[position], dtype=np.int64))
    a = np.minimum(np.maximum(a, duration), L)
    rows = np.arange(n)
    for o in range(duration):                 # ages a, a-1, ..., a-d+1
        D[rows, L - (a - o)] += mags
    return D


def single_replacement_ranks(sc: np.ndarray, pos: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rank (1 = best, stable order) of item pos[j] after its score is replaced
    by v[j], all other scores unchanged.  Vectorised over j."""
    ar = np.arange(len(sc))
    gt = (sc[None, :] > v[:, None]).sum(1) - (sc[pos] > v)
    eq = ((sc[None, :] == v[:, None]) & (ar[None, :] < pos[:, None])).sum(1)
    return 1 + gt + eq


def ranks_of(sc: np.ndarray) -> np.ndarray:
    r = np.empty(len(sc))
    r[stable_order(sc)] = np.arange(1, len(sc) + 1)
    return r


def noise_realisations(base: np.ndarray, seed: int, k: int) -> dict:
    """Noisy copies of the 64-slot matrix of all items (shared by all methods)."""
    out = {'noise_poisson': np.random.default_rng([seed, k, 2]).poisson(base).astype(np.float64)}
    ms = (base ** 2).mean(axis=1)
    for tag, snr in ((3, 10), (4, 0)):
        sig = np.sqrt(ms / 10 ** (snr / 10))
        z = np.random.default_rng([seed, k, tag]).standard_normal(base.shape)
        out[f'noise_gauss_snr{snr}'] = np.maximum(0.0, base + z * sig[:, None])
    return out


class Windows:
    """Eligibility, pool and targets per window - identical in every process."""

    def __init__(self, fe: ProtocolV5Evaluator, methods: dict, stride: int = 1):
        self.fe, self.methods, self.stride = fe, methods, int(stride)

    def __iter__(self):
        fe = self.fe
        cum = np.zeros(len(fe.items))
        for k in range(fe.num_slots + 1):
            if k > 0:
                cum += fe.V[:, k - 1]
            if k < 32 or not fe.any_row[k] or k % self.stride:
                continue
            elig, ok = {}, True
            for key, fm in self.methods.items():
                lo, hi = fe.window_bounds(k, fm.window_slots)
                if fe.any_cum[hi] - fe.any_cum[lo] == 0:
                    ok = False
                    break
                m = (fe.C[:, hi] - fe.C[:, lo] >= fm.min_obs)
                if fe.causal_universe:
                    m &= cum >= fe.universe_min_count
                idx = np.where(m)[0]
                if len(idx) < 2:
                    ok = False
                    break
                elig[key] = idx
            if not ok:
                continue
            pool = None
            for idx in elig.values():
                pool = idx if pool is None else np.intersect1d(pool, idx)
            lo = max(k - REF, 0)
            m64 = fe.V[pool, lo:k].mean(axis=1)
            nz = np.where(m64 > 0)[0]
            if len(nz) == 0:
                continue
            cand = nz[m64[nz] <= np.median(m64[nz])]
            yield k, elig, pool, cand, m64, lo


def _finite_mean(x: np.ndarray) -> float:
    """Mean over finite values (a noisy series can become all zeros -> log 0)."""
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def _diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        return a - b


def targets_for(cand: np.ndarray, seed: int, k: int) -> np.ndarray:
    rng = np.random.default_rng([seed, k])
    return rng.choice(cand, min(N_TARGETS, len(cand)), replace=False)


# =============================================================================
# One method
# =============================================================================
def run_methods(fe, windows: Windows, keys, methods, seeds, verbose=True,
                start_after: int = -1, flush=None, flush_every: int = 2000):
    """One pass over the windows for all requested (method, W).  Targets,
    random ages, spike matrices and noise are built once per (window, seed)
    and shared, so the result of a method does not depend on which other
    methods run in the same process (checked in the unit test)."""
    trend = make_v5_wspi(window=64, use_R=False, use_WE=False)
    recs = {key: [] for key in keys}
    n_done = {key: 0 for key in keys}
    tm = {key: 0.0 for key in keys}
    t_all = time.time()
    n_since = 0
    for k, elig, pool, cand, m64, lo in windows:
        if k <= start_after:
            continue
        L = k - lo
        base_all = fe.V[:, lo:k]
        shared = []
        for s in seeds:
            tg = targets_for(cand, s, k)
            ages = np.random.default_rng([s, k, 1]).integers(1, L + 1, size=len(tg))
            Xt_ref = fe.V[pool[tg], lo:k]                      # targets, reference span
            spiked = [Xt_ref + spike_matrix(L, np.maximum(sz * m64[tg], 1.0), p, d, ages)
                      for (_, sz, p, d) in SPIKES]
            shared.append((s, tg, Xt_ref, spiked, noise_realisations(base_all, s, k)))
        for key in keys:
            t1 = time.time()
            name, W = key
            fm = methods[key]
            is_wspi = name == 'WSPI'
            idx = elig[key]
            X = fe.V[idx, max(k - W, 0):k]
            sc = fe._safe_score(fm.scorer, X)
            rk = ranks_of(sc)
            order = stable_order(sc)
            top_clean = set(order[:10].tolist())
            rec = {'window_id': k, 'num_items': len(idx), 'npool': len(pool),
                   'ndcg@10_clean': ndcg_stable(order, fe.V[idx, k], 10)}
            acc = {c: {'dr': [], 'sdr': [], 'up': [], 'top10': [], 'dlogmu': [], 'dexpo': []}
                   for c in CONDITIONS}
            for s, tg, Xt_ref, spiked, noisy in shared:
                rec['ntargets'] = len(tg)
                pos = np.searchsorted(idx, pool[tg])
                P = np.concatenate([Z[:, -W:] for Z in spiked])
                ns = fe._safe_score(fm.scorer, P).reshape(len(SPIKES), len(tg))
                if is_wspi:
                    mu_c = trend(Xt_ref[:, -64:])
                    mu_n = trend(P).reshape(len(SPIKES), len(tg))
                    ls_c = np.log(sc[pos])
                for ci, (c, *_ ) in enumerate(SPIKES):
                    sg = single_replacement_ranks(sc, pos, ns[ci]) - rk[pos]
                    a = acc[c]
                    d_abs = np.abs(sg).mean()
                    a['dr'].append(d_abs); a['sdr'].append(sg.mean())
                    a['up'].append((sg < 0).mean())
                    rec[f'dr_{c}_s{s}'] = float(d_abs)
                    if is_wspi:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            dmu = np.log(mu_n[ci]) - np.log(mu_c)
                            dls = np.log(ns[ci]) - ls_c
                        a['dlogmu'].append(_finite_mean(dmu)); a['dexpo'].append(_finite_mean(_diff(dls, dmu)))
                for c, *_ in NOISES:
                    Xn = noisy[c][idx][:, -W:] if L >= W else noisy[c][idx]
                    scn = fe._safe_score(fm.scorer, Xn)
                    rkn = ranks_of(scn)
                    sg = rkn[pos] - rk[pos]
                    t2 = set(stable_order(scn)[:10].tolist())
                    a = acc[c]
                    d_abs = np.abs(sg).mean()
                    a['dr'].append(d_abs); a['sdr'].append(sg.mean())
                    a['up'].append((sg < 0).mean())
                    a['top10'].append(len(top_clean & t2) / len(top_clean | t2))
                    rec[f'dr_{c}_s{s}'] = float(d_abs)
                    if is_wspi:
                        tn = noisy[c][pool[tg]][:, -64:]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            dmu = np.log(trend(tn)) - np.log(mu_c)
                            dls = np.log(scn[pos]) - ls_c
                        a['dlogmu'].append(_finite_mean(dmu)); a['dexpo'].append(_finite_mean(_diff(dls, dmu)))
            for c in CONDITIONS:
                for stat, v in acc[c].items():
                    if v:
                        rec[f'{stat}_{c}'] = float(np.mean(v))
            recs[key].append(rec)
            tm[key] += time.time() - t1
        n_since += 1
        if flush is not None and n_since >= flush_every:
            for key in keys:
                n_done[key] += len(recs[key])
            flush(recs)
            recs = {key: [] for key in keys}
            n_since = 0
            if verbose:
                print(f'[T3.5] checkpoint at window {k}  ({time.time() - t_all:.0f}s)', flush=True)
    for key in keys:
        n_done[key] += len(recs[key])
    if flush is not None:
        flush(recs)
    if verbose:
        for key in keys:
            print(f'[T3.5] {key[0]:<12} W={key[1]:<3} windows={n_done[key]:<6} {tm[key]:8.1f}s',
                  flush=True)
        print(f'[T3.5] total {time.time() - t_all:.1f}s', flush=True)
    return recs, tm


def run_method(fe, windows, key, fm, seeds, verbose=True):
    """Single-method wrapper (tests and profiling): returns the window table."""
    recs, _ = run_methods(fe, windows, [key], {key: fm}, seeds, verbose)
    return pd.DataFrame(recs[key])


def seed_table(df: pd.DataFrame, seeds) -> pd.DataFrame:
    """Per seed and condition: mean |rank change| over the rows of df."""
    return pd.DataFrame([{'seed': s, 'condition': c, 'n_windows': len(df),
                          'dr_mean': df[f'dr_{c}_s{s}'].mean()}
                         for s in seeds for c in CONDITIONS if f'dr_{c}_s{s}' in df.columns])


def run(data, min_obs, out, causal_universe, names=None, seeds=SEEDS, resume=False,
        stride=1, checkpoint=2000):
    fe = ProtocolV5Evaluator.from_csv(data, dataset_min_obs=min_obs, seed=42,
                                      causal_universe=causal_universe, verbose=False)
    methods = all_methods()
    windows = Windows(fe, methods, stride)
    rdir = out / 'robustness'
    rdir.mkdir(parents=True, exist_ok=True)
    md = out / 'metadata'
    md.mkdir(exist_ok=True)
    mf = md / 'robustness_run.json'
    old = json.loads(mf.read_text(encoding='utf-8')) if mf.exists() else {}
    for field, val in (('stride', stride), ('seeds', list(seeds)),
                       ('causal_universe', causal_universe)):
        if field in old and old[field] != val:
            raise SystemExit(f'{field}={val} differs from the existing run in {out} '
                             f'({old[field]}); use another --out')

    keys, start_after = [], None
    for key in [(n, 7) for n in BASELINES] + [(n, 64) for n in ORDER]:
        if names and key[0] not in names:
            continue
        tag = f'{key[0]}_W{key[1]:03d}'
        final, part = rdir / f'{tag}.csv', rdir / f'{tag}.partial.csv'
        if final.exists():
            if resume:
                print(f'[T3.5] {tag} complete, skipped (--resume)')
                continue
            final.unlink()
        last = -1
        if part.exists():
            if resume:
                pk = pd.read_csv(part, usecols=['window_id'])['window_id']
                last = int(pk.max()) if len(pk) else -1
            else:
                part.unlink()
        start_after = last if start_after is None else min(start_after, last)
        keys.append(key)
    rt = {}
    if keys:
        # all keys of this pass restart from the same window
        for key in keys:
            part = rdir / f'{key[0]}_W{key[1]:03d}.partial.csv'
            if part.exists():
                d = pd.read_csv(part, float_precision='round_trip')
                d[d['window_id'] <= start_after].to_csv(part, index=False, encoding='utf-8')
        if start_after >= 0:
            print(f'[T3.5] resuming after window {start_after}')

        def flush(recs):
            for key, rows in recs.items():
                if not rows:
                    continue
                part = rdir / f'{key[0]}_W{key[1]:03d}.partial.csv'
                d = pd.DataFrame(rows)
                if part.exists():
                    cols = pd.read_csv(part, nrows=0).columns
                    d = d.reindex(columns=cols)
                d.to_csv(part, mode='a', header=not part.exists(), index=False,
                         encoding='utf-8')

        _, tm = run_methods(fe, windows, keys, methods, seeds, start_after=start_after,
                            flush=flush, flush_every=checkpoint)
        for key in keys:
            tag = f'{key[0]}_W{key[1]:03d}'
            (rdir / f'{tag}.partial.csv').replace(rdir / f'{tag}.csv')
            rt[tag] = round(tm[key], 1)
    meta = dict(old)
    meta.update(task='T3.5 extended robustness (E4)', protocol='v5', stride=stride,
                window_rule='k >= 32, row at k, k % stride == 0, every method has >= 2 items',
                data=str(data), dataset_min_obs=min_obs, causal_universe=causal_universe,
                seeds=list(seeds), n_targets=N_TARGETS, reference_slots=REF,
                spikes=[dict(name=a, size=b, position=c, duration=d) for a, b, c, d in SPIKES],
                position_age=POS_AGE, noises=[dict(name=a, kind=b, snr_db=c) for a, b, c in NOISES],
                candidates='pool items with mean_64 > 0 and mean_64 <= median of non-zero mean_64',
                pool='items eligible for all methods of both configurations',
                spike_magnitude='max(size * mean_64, 1) per slot',
                rank="position in the method's own eligible set, stable order",
                slot_minutes=fe.slot.total_seconds() / 60, num_items=len(fe.items),
                numpy=np.__version__, python=platform.python_version(),
                host=platform.platform(), updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    meta.setdefault('runtime_seconds', {}).update(rt)
    mf.write_text(json.dumps(meta, indent=2), encoding='utf-8')


# =============================================================================
# Audit of the V4/V5 test used in the paper tables (protocol_v5._robustness_v5)
# =============================================================================
AUDIT_METHODS = [('AF', 7), ('RRD', 7), ('DTCWT+AF', 64), ('WSPI', 64), ('RRD', 64)]


def audit_current_test(data, min_obs, causal_universe, ref_default: Path, scenario: str):
    """Replays the target choice and spike of ProtocolV5Evaluator._robustness_v5
    (per-method eligible set, spike = 10 x mean of the method's own W-slot
    series, last slot, one slot) and reports, on the windows >= 32 common to
    the audited methods: target overlap between methods, spike size, share of
    zero slots, and a control of the replayed Delta Rank against the T1.5 run."""
    from evaluation.protocol_v5 import rank_distortion_stable
    fe = ProtocolV5Evaluator.from_csv(data, dataset_min_obs=min_obs, seed=42,
                                      causal_universe=causal_universe, verbose=False)
    ms = all_methods()
    per, tsets = {}, {}
    for key in AUDIT_METHODS:
        fm = ms[key]
        rng = np.random.RandomState(42)
        W = fm.window_slots
        cum = np.zeros(len(fe.items))
        rows = []
        for k in range(fe.num_slots + 1):
            if k > 0:
                cum += fe.V[:, k - 1]
            lo, hi = fe.window_bounds(k, W)
            if fe.any_cum[hi] - fe.any_cum[lo] == 0 or not fe.any_row[k]:
                continue
            ok = fe.C[:, hi] - fe.C[:, lo] >= fm.min_obs
            if fe.causal_universe:
                ok &= cum >= fe.universe_min_count
            idx = np.where(ok)[0]
            if len(idx) < 2:
                continue
            X = fe._matrix(idx, k, W)
            sc = fe._safe_score(fm.scorer, X)
            ni, means = X.shape[0], X.mean(axis=1)
            nzi = np.where(means > 0)[0]
            if ni < fe.sample_size:
                tg = list(range(ni))
            elif len(nzi) < fe.sample_size:
                tg = nzi.tolist()
            else:
                thr = np.percentile(means[nzi], 50)
                cand = [i for i in nzi if means[i] <= thr]
                tg = (rng.choice(cand, fe.sample_size, replace=False).tolist()
                      if len(cand) > fe.sample_size else cand)
            if not tg:
                continue
            T = X[tg].copy()
            sp = np.maximum(T.mean(axis=1) * fe.spike_multiplier, 1.0)
            T[:, -1] += sp
            ns = fe._safe_score(fm.scorer, T)
            dr = []
            for j, t in enumerate(tg):
                noisy = sc.copy()
                noisy[t] = float(ns[j])
                dr.append(rank_distortion_stable(sc, noisy, t))
            tsets[(key, k)] = set(idx[tg].tolist())
            rows.append({'window_id': k, 'num_items': ni, 'dr': float(np.mean(dr)),
                         'spike_median': float(np.median(sp)),
                         'zero_share_targets': float(np.mean(X[tg] == 0))})
        per[key] = pd.DataFrame(rows).set_index('window_id')
    ids = sorted(set.intersection(*[set(d.index) for d in per.values()]))
    ids = [k for k in ids if k >= 32]
    out = []
    for key, d in per.items():
        r = {'scenario': scenario, 'method': key[0], 'window': key[1], 'n_windows': len(ids),
             'mean_num_items': d.loc[ids, 'num_items'].mean(),
             'median_spike': float(np.median(d.loc[ids, 'spike_median'])),
             'zero_share_targets': d.loc[ids, 'zero_share_targets'].mean(),
             'replayed_delta_rank': d.loc[ids, 'dr'].mean()}
        ref = ref_default / scenario / 'protocol' / f'{key[0]}_protocol.csv'
        if key[1] == (7 if key[0] in BASELINES else 64) and ref.exists():
            p = pd.read_csv(ref).set_index('window_id')
            c = d.index.intersection(p.index)
            r['control_max_abs_diff_vs_T1.5'] = float(np.max(np.abs(
                d.loc[c, 'dr'] - p.loc[c, 'robustness_distortion'])))
        for other in AUDIT_METHODS:
            if other == key:
                continue
            j = [len(tsets[(key, k)] & tsets[(other, k)]) / len(tsets[(key, k)] | tsets[(other, k)])
                 for k in ids]
            r[f'target_jaccard_vs_{other[0]}_W{other[1]:03d}'] = float(np.mean(j))
        out.append(r)
    return pd.DataFrame(out)


# =============================================================================
# Collect, control, statistics
# =============================================================================
def load_scenario(sdir: Path):
    res = {}
    for f in sorted((sdir / 'robustness').glob('*_W[0-9][0-9][0-9].csv')):
        name, w = f.stem.rsplit('_W', 1)
        res[(name, int(w))] = pd.read_csv(f)
    return res


def collect(root: Path, ref_default: Path, ref_eq64: Path):
    rows, ctrl = [], []
    for sdir in sorted(p for p in root.iterdir() if (p / 'robustness').is_dir()):
        scen = sdir.name
        res = load_scenario(sdir)
        for config in ('default', 'eq64'):
            mem = [m for m in config_members(config) if m in res]
            if not mem:
                continue
            ids = sorted(set.intersection(*[set(res[m]['window_id']) for m in mem]))
            for m in mem:
                d = res[m].set_index('window_id').loc[ids]
                sd = seed_table(d, SEEDS)
                for c in CONDITIONS:
                    r = {'scenario': scen, 'config': config, 'method': m[0], 'window': m[1],
                         'condition': c, 'n_windows': len(ids)}
                    for stat in ('dr', 'sdr', 'up', 'top10', 'dlogmu', 'dexpo'):
                        col = f'{stat}_{c}'
                        if col in d.columns:
                            r[f'{stat}_mean'] = d[col].mean()
                            if stat == 'dr':
                                r['dr_sd'] = d[col].std()
                    if len(sd):
                        v = sd.loc[sd['condition'] == c, 'dr_mean']
                        r['dr_seed_min'], r['dr_seed_max'] = v.min(), v.max()
                        r['dr_seed_sd'] = v.std()
                    rows.append(r)
        # control: clean NDCG@10 and num_items against the reference runs
        for (name, w), d in res.items():
            if w == 7 or name in WAVELETS:
                ref = ref_default / scen / 'protocol' / f'{name}_protocol.csv'
            else:
                ref = ref_eq64 / scen / 'W064' / 'protocol' / f'{name}_protocol.csv'
            r = {'scenario': scen, 'method': name, 'window': w, 'reference': str(ref)}
            if not ref.exists():
                r['status'] = 'missing reference'
            else:
                p = pd.read_csv(ref).set_index('window_id')
                x = d.set_index('window_id')
                common = x.index.intersection(p.index)
                r['n_windows'] = len(x)
                r['n_matched'] = len(common)
                r['max_abs_diff_ndcg10'] = float(np.max(np.abs(
                    x.loc[common, 'ndcg@10_clean'] - p.loc[common, 'ndcg@10'])))
                r['num_items_equal'] = bool((x.loc[common, 'num_items'] ==
                                             p.loc[common, 'num_items']).all())
                r['status'] = ('equal' if len(common) == len(x) and r['num_items_equal']
                               and r['max_abs_diff_ndcg10'] < 1e-12 else 'DIFFERENT')
            ctrl.append(r)
    pd.DataFrame(rows).to_csv(root / 'robustness_summary.csv', index=False, encoding='utf-8')
    # main-table check: the V4/V5 test (per-method targets, robustness_distortion of
    # the T1.5 runs) against the common test at 10x / last / 1 slot, same windows
    mrows = []
    for sdir in sorted(p for p in root.iterdir() if (p / 'robustness').is_dir()):
        res = load_scenario(sdir)
        mem = [m for m in config_members('default') if m in res]
        if not mem:
            continue
        ids = sorted(set.intersection(*[set(res[m]['window_id']) for m in mem]))
        part = []
        for m in mem:
            ref = ref_default / sdir.name / 'protocol' / f'{m[0]}_protocol.csv'
            if not ref.exists():
                continue
            p = pd.read_csv(ref).set_index('window_id')
            part.append({'scenario': sdir.name, 'method': m[0], 'n_windows': len(ids),
                         'main_test_mean': p.loc[ids, 'robustness_distortion'].mean(),
                         'common_test_mean': res[m].set_index('window_id').loc[ids, 'dr_size10'].mean()})
        if part:
            d = pd.DataFrame(part)
            d['main_test_rank'] = d['main_test_mean'].rank(method='min').astype(int)
            d['common_test_rank'] = d['common_test_mean'].rank(method='min').astype(int)
            mrows.append(d)
    if mrows:
        pd.concat(mrows).to_csv(root / 'robustness_main_check.csv', index=False, encoding='utf-8')
    c = pd.DataFrame(ctrl)
    c.to_csv(root / 'robustness_control.csv', index=False, encoding='utf-8')
    print(c[['scenario', 'method', 'window', 'status']].to_string(index=False))
    print(f'\n{root / "robustness_summary.csv"}: {len(rows)} rows')


def stats(root: Path, out_root: Path, B: int, only=None):
    sys.path.insert(0, str(ROOT / 'tools'))
    import stats_report as sr
    for sdir in sorted(p for p in root.iterdir() if (p / 'robustness').is_dir()):
        scen = sdir.name
        if only and scen not in only:
            continue
        block, block_sens = DATASETS[scen][2], DATASETS[scen][3]
        mf = sdir / 'metadata' / 'robustness_run.json'
        stride = json.loads(mf.read_text(encoding='utf-8')).get('stride', 1) if mf.exists() else 1
        block = max(1, block // stride)                 # block length in windows
        block_sens = max(1, block_sens // stride) if block_sens else None
        res = load_scenario(sdir)
        for config in ('default', 'eq64'):
            mem = [m for m in config_members(config) if m in res]
            if ('WSPI', 64) not in mem or len(mem) < 2:
                continue
            for c in CONDITIONS:
                tmp = Path(tempfile.mkdtemp(prefix='t35_'))
                try:
                    (tmp / 'protocol').mkdir()
                    for m in mem:
                        d = res[m]
                        p = pd.DataFrame({'window_id': d['window_id'],
                                          'robustness_distortion': d[f'dr_{c}']})
                        if f'top10_{c}' in d.columns:
                            p['top10_overlap'] = d[f'top10_{c}']
                        p.to_csv(tmp / 'protocol' / f'{m[0]}_protocol.csv', index=False)
                    o = out_root / scen / config / c
                    sr.analyse(tmp, o, block, block_sens, 'WSPI', B, 42, None)
                    print(f'[stats] {scen}/{config}/{c}')
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
    sr.collect(out_root)


def main():
    ap = argparse.ArgumentParser(description='T3.5 extended robustness (E4), protocol V5')
    ap.add_argument('--data')
    ap.add_argument('--min-obs', type=int)
    ap.add_argument('--out')
    ap.add_argument('--causal-universe', action='store_true')
    ap.add_argument('--methods', nargs='*', default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--checkpoint', type=int, default=2000,
                    help='write the partial CSVs every N windows')
    ap.add_argument('--stride', type=int, default=1,
                    help='evaluate every stride-th window (taxi 5-min: 12, one window per hour)')
    ap.add_argument('--collect', help='root with <scenario>/robustness/*.csv')
    ap.add_argument('--ref-default', default='results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5')
    ap.add_argument('--ref-eq64', default='results/revision_v5/T2.2_window_sweep')
    ap.add_argument('--stats', help='root with <scenario>/robustness/*.csv')
    ap.add_argument('--stats-out')
    ap.add_argument('--scenarios', nargs='*', default=None, help='--stats: only these')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--audit', help='scenario name: audit the V4/V5 test of the paper tables '
                                    'and append to <audit-out>/current_test_audit.csv')
    ap.add_argument('--audit-out', default='results/revision_v5/T3.5_robustness')
    a = ap.parse_args()

    def absp(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    if a.collect:
        collect(absp(a.collect), absp(a.ref_default), absp(a.ref_eq64))
        return
    if a.audit:
        if not (a.data and a.min_obs):
            ap.error('--audit needs --data and --min-obs')
        df = audit_current_test(absp(a.data), a.min_obs, a.causal_universe,
                                absp(a.ref_default), a.audit)
        o = absp(a.audit_out)
        o.mkdir(parents=True, exist_ok=True)
        f = o / 'current_test_audit.csv'
        if f.exists():
            old = pd.read_csv(f)
            df = pd.concat([old[old['scenario'] != a.audit], df], ignore_index=True)
        df.to_csv(f, index=False, encoding='utf-8')
        print(df.round(3).to_string(index=False))
        return
    if a.stats:
        if not a.stats_out:
            ap.error('--stats needs --stats-out')
        stats(absp(a.stats), absp(a.stats_out), a.B, a.scenarios)
        return
    if not (a.data and a.min_obs and a.out):
        ap.error('--data, --min-obs and --out are required')
    if not a.causal_universe:
        print('WARNING: --causal-universe not given (the paper uses it)')
    run(absp(a.data), a.min_obs, absp(a.out), a.causal_universe, a.methods,
        tuple(a.seeds), a.resume, a.stride, a.checkpoint)
    print(f'Saved to {absp(a.out)}')


if __name__ == '__main__':
    main()
