r"""
Responsiveness to genuine entries into the true Top-10 (revision-srep-v5, task T2.4 / E11)
=========================================================================================
Answers reviewer comment R4.12: does WSPI suppress or delay genuine rises in
popularity?  No existing module is changed; the evaluator re-uses the matrix,
the causal item catalogue and the eligibility rule of ``protocol_v5``.

Design (approved by Sajjad, 25 Sep 2026, chat 7)
-----------------------------------------------
Ground truth.  At slot k the true ranking orders the catalogue items by their
real count in slot k (the evaluation target, horizon 1 slot).  Catalogue =
the causal item catalogue of T1.5 (total count before slot k >= dataset
min_obs).  Ties: stable sort, fixed item order (same rule as protocol V5).
The truth does NOT depend on any method.

Event ("entry").  Item i enters at slot t0 when
  - it is in the true Top-10 in the L slots t0 .. t0+L-1   (default L = 6),
  - it is outside the true Top-``pre`` in each of the P slots before t0
    (default P = 6, pre = 10: main variant),
  - t0 >= first_window (default 32, as every table of the revision).
``run_len`` R = number of consecutive slots from t0 on in the true Top-10.
``strict`` = also outside the true Top-20 in the P slots before t0
(sensitivity variant, CSV / SI only).  Slots without any data row are
treated as outside the Top-10.

Delay.  Window k of a method ranks the items for slot k from the slots
[k-W, k).  delay d = min{k - t0 : t0 <= k <= t0+R-1, i in the method's
Top-10 at window k}.  d = 0: the method already had the item in its Top-10
before seeing any slot of the entry; d = 1 is the smallest delay of a purely
reactive method.  No such k: ``miss`` (the method never shows the item while
it is truly in the Top-10).  ``delay_restricted`` = d, or R for a miss.
An item that is not eligible for the method at window k (fewer than
min_obs observed rows in the slice) is "not in the Top-10"; the number of
such windows during the run is recorded (``n_ineligible_run``).

Statistics.  WSPI against every other method on ``delay_restricted``, paired
by event.  Clusters = non-overlapping time blocks of the entry slot t0
(block length as in T1.6: YouTube 24 slots, taxi one week).  95 % CI of the
mean paired difference from a cluster bootstrap (B resamples, seed); Wilcoxon
signed-rank on cluster means; Holm within one scenario x configuration x
variant.  Orientation: positive effect = WSPI faster.

Stability failures (item 4 of E11).  For every window common to all methods
of a configuration (k >= first_window): RSI@10 of every method and the truth
change the window could see (``truth_rsi_seen@10``: Jaccard of the true Top-10
of the two previous evaluated slots).  For each method: Spearman correlation
of its RSI@10 with the seen truth change, the share of windows where RSI@10 of
WSPI is lower, and the seen truth change in those windows versus the others.

Control.  NDCG@10 and RSI@10 recomputed from the stored Top-10 lists are
compared, window by window, with an existing protocol run of the same
configuration (``T1.5_causal_universe`` for ``default``,
``T2.2_window_sweep/W064`` for ``equal64``).

Author: Sajjad (with assistance), September 2026
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .fast_evaluator import FastMethod
from .metrics import calculate_rsi
from .protocol_v5 import ProtocolV5Evaluator, build_v5_methods, ndcg_stable, stable_order

TOPK = 10
NOT_RANKED = np.iinfo(np.uint16).max
DEFAULT_METHODS = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF',
                   'DWT+AF', 'DTCWT+AF', 'WSPI']
CONFIGS = ('default', 'equal64')


def build_config(config: str, level: int = 3) -> Dict[str, FastMethod]:
    """default: baselines 7, wavelet-based 64 (table 1 of the paper).
    equal64: every method with a 64-slot window (table 2; same settings as
    the N = 64 point of the T2.2 sweep)."""
    if config == 'default':
        return build_v5_methods(level=level)
    if config == 'equal64':
        from .sweep_methods import build_sweep_methods
        return build_sweep_methods(64, level=level, names=DEFAULT_METHODS)
    raise ValueError(config)


# =============================================================================
# Statistics helpers (same conventions as tools/stats_report.py)
# =============================================================================
def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    running = 0.0
    for i, idx in enumerate(ok[np.argsort(p[ok], kind='stable')]):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=float)
    if np.count_nonzero(d) == 0:
        return 1.0
    return float(wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)


def rank_biserial(d: np.ndarray) -> float:
    from scipy.stats import rankdata
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return float('nan')
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def cluster_bootstrap_mean(d: np.ndarray, cluster: np.ndarray, B: int, seed: int):
    """Percentile 95 % CI of the event-weighted mean of d, resampling whole
    clusters with replacement.  Deterministic in (seed, number of clusters)."""
    ids, inv = np.unique(cluster, return_inverse=True)
    nc = len(ids)
    s = np.bincount(inv, weights=d, minlength=nc)
    c = np.bincount(inv, minlength=nc).astype(float)
    rng = np.random.default_rng([seed, nc])
    idx = rng.integers(0, nc, size=(B, nc))
    boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), nc


def cluster_means(d: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    ids, inv = np.unique(cluster, return_inverse=True)
    return np.bincount(inv, weights=d) / np.bincount(inv)


# =============================================================================
# Evaluator
# =============================================================================
class ResponsivenessEvaluator(ProtocolV5Evaluator):
    """Protocol-V5 matrix + truth ranks + per-window Top-10 of every method."""

    def __init__(self, data: pd.DataFrame, dataset_min_obs: int,
                 causal_universe: bool = True, first_window: int = 32, **kw):
        kw.pop('robustness', None)
        super().__init__(data, dataset_min_obs, robustness=False,
                         causal_universe=causal_universe, **kw)
        self.first_window = first_window
        self.nk = self.num_slots + 1                       # slots 0 .. num_slots
        n = len(self.items)
        if self.causal_universe:
            before = np.zeros((n, self.nk))
            np.cumsum(self.V[:, :self.nk - 1], axis=1, out=before[:, 1:])
            self.catalogue = before >= self.universe_min_count
        else:
            self.catalogue = np.ones((n, self.nk), dtype=bool)
        self.slot_ok = self.any_row[:self.nk].copy()
        self._truth()

    # -------------------------------------------------------------------------
    def _truth(self):
        """rank[i, k]: 1-based true rank in slot k (NOT_RANKED if not in the
        catalogue or no data in the slot); top[k]: the true Top-10 (item rows)."""
        n = len(self.items)
        self.truth_rank = np.full((n, self.nk), NOT_RANKED, dtype=np.uint16)
        self.truth_top = np.full((self.nk, TOPK), -1, dtype=np.int32)
        for k in range(self.nk):
            if not self.slot_ok[k]:
                continue
            idx = np.where(self.catalogue[:, k])[0]
            if len(idx) == 0:
                continue
            o = idx[stable_order(self.V[idx, k])]
            self.truth_rank[o, k] = np.minimum(np.arange(1, len(o) + 1), NOT_RANKED - 1)
            t = o[:TOPK]
            self.truth_top[k, :len(t)] = t

    # -------------------------------------------------------------------------
    def find_events(self, L: int = 6, P: int = 6, pre: int = 10,
                    pre_strict: int = 20) -> pd.DataFrame:
        inT = self.truth_rank <= TOPK
        rows = []
        t_min = max(self.first_window, P)
        for i in range(len(self.items)):
            x = inT[i]
            starts = np.where(x[1:] & ~x[:-1])[0] + 1
            for t0 in starts:
                if t0 < t_min or t0 + L > self.nk or not x[t0:t0 + L].all():
                    continue
                pr = self.truth_rank[i, t0 - P:t0]
                if not (pr > pre).all():
                    continue
                r = L
                while t0 + r < self.nk and x[t0 + r]:
                    r += 1
                run = self.V[i, t0:t0 + r]
                pv = self.V[i, t0 - P:t0]
                rows.append({
                    'item_id': self.items[i], 'item_row': i, 't0': int(t0),
                    'timestamp': int((self.start + int(t0) * self.slot).timestamp() * 1000),
                    'run_len': int(r), 'run_hits_end': bool(t0 + r >= self.nk),
                    'strict': bool((pr > pre_strict).all()),
                    'pre_best_rank': int(pr.min()) if pr.min() < NOT_RANKED else -1,
                    'pre_mean_count': float(pv.mean()), 'run_mean_count': float(run.mean()),
                    'run_peak_count': float(run.max()),
                    'count_t0': float(self.V[i, t0])})
        ev = pd.DataFrame(rows)
        if len(ev):
            ev = ev.sort_values(['t0', 'item_row']).reset_index(drop=True)
        ev.insert(0, 'event_id', np.arange(len(ev)))
        return ev

    # -------------------------------------------------------------------------
    def _eligible(self, fm: FastMethod, k: int, cum: Optional[np.ndarray]):
        """Replica of ProtocolV5Evaluator.run_method eligibility at window k.
        Returns None when run_method would skip the window."""
        lo, hi = self.window_bounds(k, fm.window_slots)
        if self.any_cum[hi] - self.any_cum[lo] == 0 or not self.any_row[k]:
            return None
        ok = (self.C[:, hi] - self.C[:, lo]) >= fm.min_obs
        if cum is not None:
            ok &= cum >= self.universe_min_count
        idx = np.where(ok)[0]
        return idx if len(idx) >= 2 else None

    def method_pass(self, fm: FastMethod):
        """One pass over all windows.  Returns
           top   (nk, 10) int32  Top-10 item rows per window (-1 = no window)
           elig  (n, nk)  bool   item eligible at window k
           win   DataFrame       window_id, ndcg@10, rsi@10 (same as protocol CSV)"""
        n = len(self.items)
        top = np.full((self.nk, TOPK), -1, dtype=np.int32)
        elig = np.zeros((n, self.nk), dtype=bool)
        cum = np.zeros(n) if self.causal_universe else None
        prev = None
        recs = []
        for k in range(self.nk):
            if cum is not None and k > 0:
                cum += self.V[:, k - 1]
            idx = self._eligible(fm, k, cum)
            if idx is None:
                continue
            lo, hi = self.window_bounds(k, fm.window_slots)
            sc = self._safe_score(fm.scorer, self.V[idx, lo:hi])
            order = stable_order(sc)
            t = idx[order[:TOPK]]
            top[k, :len(t)] = t
            elig[idx, k] = True
            tl = [int(v) for v in t]
            recs.append({'window_id': k,
                         'ndcg@10': ndcg_stable(order, self.V[idx, k], TOPK),
                         'rsi@10': float('nan') if prev is None else calculate_rsi(prev, tl)})
            prev = tl
        return top, elig, pd.DataFrame(recs)

    def method_ranks(self, fm: FastMethod, item_row: int, k_lo: int, k_hi: int) -> pd.DataFrame:
        """Rank of one item in the method's ranking for windows k_lo..k_hi
        (NaN when the item is not eligible or the window is skipped)."""
        n = len(self.items)
        cum = None
        if self.causal_universe:
            cum = self.V[:, :k_lo].sum(axis=1) if k_lo > 0 else np.zeros(n)
        out = []
        for k in range(k_lo, k_hi + 1):
            if cum is not None and k > k_lo:
                cum = cum + self.V[:, k - 1]
            idx = self._eligible(fm, k, cum)
            rank, el = np.nan, False
            if idx is not None and item_row in set(idx.tolist()):
                lo, hi = self.window_bounds(k, fm.window_slots)
                sc = self._safe_score(fm.scorer, self.V[idx, lo:hi])
                pos = np.where(idx[stable_order(sc)] == item_row)[0][0]
                rank, el = float(pos + 1), True
            out.append({'window_id': k, 'rank': rank, 'eligible': el})
        return pd.DataFrame(out)

    # -------------------------------------------------------------------------
    @staticmethod
    def delays(events: pd.DataFrame, top: np.ndarray, elig: np.ndarray,
               method: str) -> pd.DataFrame:
        rows = []
        for e in events.itertuples(index=False):
            i, t0, R = e.item_row, e.t0, e.run_len
            hit = (top[t0:t0 + R] == i).any(axis=1)
            el = elig[i, t0:t0 + R]
            d = int(np.argmax(hit)) if hit.any() else None
            rows.append({'event_id': e.event_id, 'method': method,
                         'detected': d is not None,
                         'delay': np.nan if d is None else d,
                         'delay_restricted': R if d is None else d,
                         'coverage': float(hit.mean()),
                         'eligible_at_t0': bool(el[0]),
                         'n_ineligible_run': int((~el).sum()),
                         'in_top_before': bool(t0 > 0 and (top[t0 - 1] == i).any())})
        return pd.DataFrame(rows)

    def truth_rsi(self) -> pd.DataFrame:
        """RSI@10 of the true Top-10 between consecutive evaluated slots."""
        recs, prev = [], None
        for k in range(self.nk):
            if not self.slot_ok[k] or self.truth_top[k, 0] < 0:
                continue
            t = [int(v) for v in self.truth_top[k] if v >= 0]
            recs.append({'window_id': k,
                         'truth_rsi@10': float('nan') if prev is None else calculate_rsi(prev, t)})
            prev = t
        return pd.DataFrame(recs)


# =============================================================================
# Summaries
# =============================================================================
def summarize_delays(ev: pd.DataFrame, dl: pd.DataFrame, variant: str) -> pd.DataFrame:
    sel = ev if variant == 'main' else ev[ev['strict']]
    d = dl[dl['event_id'].isin(sel['event_id'])].merge(
        sel[['event_id', 'run_len', 'item_id']], on='event_id')
    rows = []
    for m, g in d.groupby('method', sort=False):
        x = np.where(g['detected'], g['delay'], np.inf)
        med = float(np.median(x)) if len(x) else np.nan
        miss = ~g['detected']
        rows.append({'variant': variant, 'method': m, 'n_events': len(g),
                     'n_items': g['item_id'].nunique(),
                     'miss_rate': float(miss.mean()) if len(g) else np.nan,
                     'median_delay': med if np.isfinite(med) else np.nan,
                     'median_undefined': bool(len(x) and not np.isfinite(med)),
                     'mean_delay_restricted': g['delay_restricted'].mean(),
                     'mean_delay_detected': g.loc[g['detected'], 'delay'].mean(),
                     'share_delay0': float((g['delay'] == 0).mean()),
                     'share_delay_le1': float((g['delay'] <= 1).mean()),
                     'share_delay_le3': float((g['delay'] <= 3).mean()),
                     'mean_coverage': g['coverage'].mean(),
                     'share_miss_ineligible_whole_run':
                         float(((g['n_ineligible_run'] == g['run_len']) & miss).sum() / miss.sum())
                         if miss.sum() else 0.0,
                     'share_ineligible_at_t0': float((~g['eligible_at_t0']).mean()),
                     'mean_run_len': g['run_len'].mean()})
    return pd.DataFrame(rows)


def paired_delays(ev: pd.DataFrame, dl: pd.DataFrame, variant: str, block: int,
                  B: int, seed: int, reference: str = 'WSPI') -> pd.DataFrame:
    sel = ev if variant == 'main' else ev[ev['strict']]
    piv = dl[dl['event_id'].isin(sel['event_id'])].pivot(
        index='event_id', columns='method', values='delay_restricted')
    piv = piv.reindex(sel['event_id'])
    cl = (sel.set_index('event_id')['t0'] // block).reindex(piv.index).to_numpy()
    rows = []
    for m in piv.columns:
        if m == reference:
            continue
        a, b = piv[reference].to_numpy(float), piv[m].to_numpy(float)
        d = a - b
        r = {'variant': variant, 'reference': reference, 'method': m, 'n_events': len(d),
             'mean_reference': a.mean() if len(d) else np.nan,
             'mean_method': b.mean() if len(d) else np.nan,
             'diff_ref_minus_method': d.mean() if len(d) else np.nan,
             'ref_faster': int((d < 0).sum()), 'ref_slower': int((d > 0).sum()),
             'ties': int((d == 0).sum()), 'block': block}
        if len(d) >= 2:
            lo, hi, nc = cluster_bootstrap_mean(d, cl, B, seed)
            cm = cluster_means(d, cl)
            r.update({'diff_ci_low': lo, 'diff_ci_high': hi, 'n_clusters': nc,
                      'p_cluster': wilcoxon_p(cm),
                      'rb_cluster_favours_ref': -rank_biserial(cm),
                      'rb_event_favours_ref': -rank_biserial(d)})
        else:
            r.update({'diff_ci_low': np.nan, 'diff_ci_high': np.nan, 'n_clusters': 0,
                      'p_cluster': np.nan, 'rb_cluster_favours_ref': np.nan,
                      'rb_event_favours_ref': np.nan})
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out['p_cluster_holm'] = holm(out['p_cluster'].to_numpy())
        sig = out['p_cluster_holm'] < 0.05
        out['verdict'] = np.where(~sig, 'n.s.', np.where(out['diff_ref_minus_method'] < 0,
                                                         'ref_faster', 'ref_slower'))
    return out


def rsi_failures(win: Dict[str, pd.DataFrame], truth: pd.DataFrame, first_window: int,
                 reference: str = 'WSPI'):
    """Window table (common windows >= first_window) and the failure summary.

    Alignment: window k ranks for slot k from the slots before k, so the last
    real change it can see is the change of the true Top-10 between the two
    previous evaluated slots.  ``truth_rsi_seen@10`` of window k = truth RSI@10
    of the previous evaluated slot (Jaccard of the true Top-10 of slots
    k-2 and k-1 when both have data).  ``truth_rsi@10`` (same slot) is kept in
    the window table for reference.  Returns (window table, summary)."""
    from scipy.stats import spearmanr
    tt = truth.sort_values('window_id').copy()
    tt['truth_rsi_seen@10'] = tt['truth_rsi@10'].shift(1)
    ids = set.intersection(*[set(w['window_id']) for w in win.values()])
    ids = sorted(k for k in ids if k >= first_window)
    t = tt.set_index('window_id').reindex(ids)
    tab = pd.DataFrame({'window_id': ids, 'truth_rsi@10': t['truth_rsi@10'].to_numpy(),
                        'truth_rsi_seen@10': t['truth_rsi_seen@10'].to_numpy()})
    for m, w in win.items():
        tab[f'rsi@10_{m}'] = w.set_index('window_id').reindex(ids)['rsi@10'].to_numpy()
    ref = tab[f'rsi@10_{reference}']
    seen = tab['truth_rsi_seen@10']

    def rho(x):
        ok = x.notna() & seen.notna()
        return float(spearmanr(x[ok], seen[ok]).correlation) if ok.sum() > 2 else np.nan

    rho_ref = rho(ref)
    rows = []
    for m in win:
        x = tab[f'rsi@10_{m}']
        r = {'reference': reference, 'method': m, 'n_windows': len(tab),
             'rsi_mean': float(x.mean()),
             'spearman_rsi_vs_truth_seen': rho(x),
             'spearman_ref_rsi_vs_truth_seen': rho_ref,
             'truth_rsi_seen_mean': float(seen.mean())}
        if m != reference:
            low = ref < x
            r.update({'share_ref_rsi_lower': float(low.mean()),
                      'share_ref_rsi_higher': float((ref > x).mean()),
                      'truth_seen_when_ref_lower': float(seen[low].mean()),
                      'truth_seen_other_windows': float(seen[~low].mean()),
                      'share_truth_seen_changed_when_ref_lower': float((seen[low] < 1).mean()),
                      'share_truth_seen_changed_other_windows': float((seen[~low] < 1).mean())})
        rows.append(r)
    return tab, pd.DataFrame(rows)


def select_examples(ev: pd.DataFrame, dl: pd.DataFrame, reference: str = 'WSPI',
                    other: str = 'AF') -> pd.DataFrame:
    """Rule fixed before seeing the results (main variant, default config):
    typical = event whose restricted-delay difference (reference - other) is
    closest to the median difference; worst = largest difference.  Ties: the
    larger run_peak_count, then the earlier t0.  The worst example is taken
    from the remaining events if it coincides with the typical one."""
    piv = dl.pivot(index='event_id', columns='method', values='delay_restricted')
    e = ev.set_index('event_id').copy()
    e['diff'] = piv[reference] - piv[other]
    if e.empty:
        return pd.DataFrame()
    med = float(np.median(e['diff']))
    e['dist'] = (e['diff'] - med).abs()
    typ = e.sort_values(['dist', 'run_peak_count', 't0'], ascending=[True, False, True]).index[0]
    rest = e.drop(index=typ)
    rows = [('typical', typ, med)]
    if len(rest):
        worst = rest.sort_values(['diff', 'run_peak_count', 't0'],
                                 ascending=[False, False, True]).index[0]
        rows.append(('worst_for_reference', worst, med))
    out = []
    for kind, eid, m in rows:
        r = e.loc[eid]
        out.append({'example': kind, 'event_id': int(eid), 'item_id': r['item_id'],
                    'item_row': int(r['item_row']), 't0': int(r['t0']),
                    'run_len': int(r['run_len']), 'run_peak_count': r['run_peak_count'],
                    f'delay_{reference}': float(piv.loc[eid, reference]),
                    f'delay_{other}': float(piv.loc[eid, other]),
                    'diff': float(r['diff']), 'median_diff_all_events': m,
                    'rule': f'{kind}: restricted delay {reference}-{other}; '
                            'typical = closest to median, worst = max; ties by peak count, then t0'})
    return pd.DataFrame(out)
