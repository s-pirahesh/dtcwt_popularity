#!/usr/bin/env python3
"""Check every number in the V5 paper and its Supplementary Information against the result files (task T4.11).

The program reads WSPI_ScientificReports.tex and WSPI_SI.tex of a built copy (the .aux file of the paper
must exist) and finds every number written with digits, from \\begin{document} to the reference list.
Each number gets one category:

  result      a number of a result; recomputed from a CSV or JSON in results/revision_v5 and compared at the
              printed precision (status ok or mismatch). The source column names the file, the row and the column.
  design      a setting of the protocol (window length, level, K=10, 95%, seed, grid values ...). Where a result
              file records the setting, it is compared too (design-ok); otherwise the reason is given.
  reference   a section, table, figure, equation or citation number; sections, tables and figures are checked
              against the .aux file, citations against the reference list.
  raw-data    a number counted from the raw data because no result file holds it (a recorded decision).

The tables of the main text are checked cell by cell, together with their marks: the triangles of
Tables 8 and 9 against the block Wilcoxon verdicts, the bold best values, the daggers of Table 12 and the
dashes of Table 14. The 24 tables that the SI reads with \\input are made by programs with their own
controls; for them the program checks that each file is identical to the output of its program.

Numbers written in words ("in all six scenarios") are not checked here.

The program only reads. It writes one CSV report (--out) with one row per number, mark and SI table file,
prints a summary, and returns 1 if any number differs, has no rule, or a rule does not match the text.

Example (Windows, from the project root, on a temporary built copy of V5/source):
  python tools\\check_paper_numbers.py --main "%TMPB%\\WSPI_ScientificReports.tex" --si "%TMPB%\\WSPI_SI.tex"
      --results results\\revision_v5 --figures-json "%RESP%\\Figures\\paper_print\\export_paper_figures_run.json"
      --si-sources "%RESP%\\Tables\\SI" "%RESP%\\Tables" --out "%RESP%\\Reports\\R29_T4.11_number_check.csv"

When a sentence of the paper changes, the rule that holds its anchor text must change with it; the
program reports such a rule as a RULE PROBLEM instead of passing it.
"""
import argparse, csv, hashlib, json, math, re, sys
from pathlib import Path

import pandas as pd

RESULTS = None          # set in main(): results/revision_v5
_CACHE = {}


def load(rel):
    """Read a result CSV (or JSON) once; rel is relative to results/revision_v5."""
    if rel not in _CACHE:
        p = RESULTS / rel
        if rel.endswith('.json'):
            _CACHE[rel] = json.loads(p.read_text(encoding='utf-8'))
        else:
            _CACHE[rel] = pd.read_csv(p)
    return _CACHE[rel]


class S:
    """A number taken from a result file, with a readable description of where it came from."""

    def __init__(self, value, desc, files=()):
        self.value = float(value)
        self.desc = desc
        self.files = tuple(files)

    def _op(self, other, sym, fn):
        o = other if isinstance(other, S) else S(other, repr(other))
        return S(fn(self.value, o.value), f'({self.desc}) {sym} ({o.desc})', self.files + o.files)

    def __mul__(self, o): return self._op(o, '*', lambda a, b: a * b)
    def __rmul__(self, o): return S(o, repr(o))._op(self, '*', lambda a, b: a * b)
    def __truediv__(self, o): return self._op(o, '/', lambda a, b: a / b)
    def __sub__(self, o): return self._op(o, '-', lambda a, b: a - b)
    def __add__(self, o): return self._op(o, '+', lambda a, b: a + b)
    def __rsub__(self, o): return S(o, repr(o))._op(self, '-', lambda a, b: a - b)
    def __radd__(self, o): return S(o, repr(o))._op(self, '+', lambda a, b: a + b)
    def __neg__(self): return S(-self.value, f'-({self.desc})', self.files)
    def __abs__(self): return S(abs(self.value), f'|{self.desc}|', self.files)


def Q(rel, col, **flt):
    """Exactly one row of result file `rel` matching the filters; value of column `col`."""
    df = load(rel)
    m = pd.Series(True, index=df.index)
    for k, v in flt.items():
        k = k.replace('__', '@').replace('_AT_', '@')
        if isinstance(v, float):
            m &= (df[k].astype(float) - v).abs() < 1e-9
        else:
            m &= df[k].astype(str) == str(v)
    rows = df[m]
    if len(rows) != 1:
        raise LookupError(f'{rel}: {len(rows)} rows for {flt}')
    f = ', '.join(f'{k}={v}' for k, v in flt.items())
    return S(rows.iloc[0][col], f'{rel} [{f}] {col}', (rel,))


def J(rel, *keys):
    """A value from a JSON result file."""
    d = load(rel)
    for k in keys:
        d = d[k]
    return S(d, f'{rel} {"/".join(map(str, keys))}', (rel,))


def smin(xs, label=''):
    xs = list(xs)
    i = min(range(len(xs)), key=lambda k: xs[k].value)
    return S(xs[i].value, f'min over {len(xs)}{label}: {xs[i].desc}', sum((x.files for x in xs), ()))


def smax(xs, label=''):
    xs = list(xs)
    i = max(range(len(xs)), key=lambda k: xs[k].value)
    return S(xs[i].value, f'max over {len(xs)}{label}: {xs[i].desc}', sum((x.files for x in xs), ()))


# --------------------------------------------------------------- tokens
NUM_RE = re.compile(
    r'(?P<sci>\d+(?:\.\d+)?\s*\\times\s*10\^\{?-?\d+\}?)'
    r'|(?P<pow>10\^\{?-?\d+\}?)'
    r'|(?P<thou>(?<![,\d])\d{1,3}(?:,\d{3})+(?!,\d)(?!\d)(?:\.\d+)?)'
    r'|(?P<plain>\d+(?:\.\d+)?)')

MASKS = [
    r'(?<!\\)%.*$',
    r'\\(?:begin\{(?:itemize|enumerate)\}|item)\[[^\]]*\]',
    r'\\rule\{[^}]*\}\{[^}]*\}',
    r'\\(?:label|ref|eqref|input|readytable|includegraphics|url|begin|end|cmidrule|addlinespace)\*?(?:\([^)]*\))?(?:\[[^\]]*\])?\{[^}]*\}(?:\{(?:@\{\}|[^{}]|\{[^}]*\})*\})?',
    r'\\(?:vspace|hspace|setlength|renewcommand|arraystretch|tabcolsep)\*?(?:\{[^}]*\}|\\\w+)*(?:\{[^}]*\})?',
    r'\\\\\[[0-9.]+(?:em|pt|ex)\]',
    r'(?:width|max width)=[0-9.]*\\?\w+',
    r'p\{[0-9.]+\\?\w+\}',
    r'\\textsuperscript\{[^}]*\}',
    r'\\shortstack',
    r'\\multicolumn\{\d+\}',
]


def masked(line):
    s = line
    for pat in MASKS:
        s = re.sub(pat, lambda m: ' ' * len(m.group(0)), s)
    return s


class Tok:
    __slots__ = ('file', 'line', 'col', 'end', 'text', 'value', 'ndec', 'sign', 'ctx', 'status',
                 'category', 'expected', 'source', 'rule', 'note', 'files')

    def __init__(self, file, line, col, end, text, value, ndec, sign, ctx):
        self.file, self.line, self.col, self.end = file, line, col, end
        self.text, self.value, self.ndec, self.sign, self.ctx = text, value, ndec, sign, ctx
        self.status = 'unchecked'
        self.category = self.expected = self.source = self.rule = self.note = ''
        self.files = ()


def parse_number(kind, t):
    if kind == 'sci':
        m = re.match(r'(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?', t)
        mant, exp = m.group(1), int(m.group(2))
        nd = len(mant.split('.')[1]) if '.' in mant else 0
        return float(mant) * 10 ** exp, nd - exp
    if kind == 'pow':
        e = int(re.match(r'10\^\{?(-?\d+)\}?', t).group(1))
        return float(10 ** e), max(0, -e)
    t2 = t.replace(',', '')
    nd = len(t2.split('.')[1]) if '.' in t2 else 0
    return float(t2), nd


def tokenize(path, tag):
    lines = Path(path).read_text(encoding='utf-8').split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith(r'\begin{document}'))
    end = next((i for i, l in enumerate(lines) if l.startswith(r'\begin{thebibliography}')), len(lines))
    toks = []
    for i in range(start + 1, end):
        raw = lines[i]
        s = masked(raw)
        for m in NUM_RE.finditer(s):
            a, b = m.span()
            prev = s[a - 1] if a > 0 else ' '
            if prev.isalpha() or prev.isdigit() or prev in '._\\':
                continue
            kind = m.lastgroup
            value, nd = parse_number(kind, m.group(0))
            # sign: a single '-' (not an en dash '--') right before the number, ignoring $ and spaces
            k = a - 1
            while k >= 0 and s[k] in ' ${':
                k -= 1
            sign = ''
            if k >= 0 and s[k] in '-+' and not (k > 0 and s[k - 1] == '-') and not (k > 0 and s[k - 1].isalnum()):
                sign = s[k]
                if sign == '-':
                    value = -value
            ctx = raw[max(0, a - 70):min(len(raw), b + 50)]
            toks.append(Tok(tag, i + 1, a, b, raw[a:b], value, nd, sign, ctx))
    return lines, toks


# ---------------------------------------------------------------- rules
def fmt(x, nd):
    if nd <= 0:
        return f'{round(x):,}' if abs(x) >= 1000 else f'{round(x)}'
    return f'{x:.{nd}f}'


def same(tok, exp_value, op='eq', scale=1.0):
    """Does the printed number agree with the value from the result file?"""
    e = exp_value * scale
    v = tok.value
    if op == 'eq':
        q = 10.0 ** (-tok.ndec)
        # printed = expected rounded to the printed decimals (half-up or half-even)
        return abs(v - e) <= q / 2 * (1 + 1e-9) + 1e-12
    if op == 'approx':   # 'about N' in the text: within 5% of the value
        return abs(v - e) <= 0.05 * abs(e)
    if op == 'gt': return e > v
    if op == 'ge': return e >= v - 1e-12
    if op == 'lt': return e < v
    if op == 'le': return e <= v + 1e-12
    raise ValueError(op)


class Spec:
    def __init__(self, src=None, op='eq', scale=1.0, cat='result', note='', design=None):
        self.src, self.op, self.scale, self.cat, self.note, self.design = src, op, scale, cat, note, design


def R(src, op='eq', scale=1.0, note=''):
    """A result number."""
    return Spec(src, op, scale, 'result', note)


def P(src, op='eq', note=''):
    """A result given in percent in the text."""
    return Spec(src, op, 100.0, 'result', note)


def D(reason, value=None):
    """A design or protocol constant, not a result; optionally verified against a result file."""
    return Spec(value, 'eq', 1.0, 'design', reason, design=True)


def Raw(reason):
    """A number checked against the raw data, with no result file (a recorded decision)."""
    return Spec(None, 'eq', 1.0, 'raw-data', reason)


def X(reason):
    """A cross-reference, citation, label or equation constant."""
    return Spec(None, 'eq', 1.0, 'reference', reason)


def apply_spec(tok, spec, rule):
    tok.rule = rule
    tok.category = spec.cat
    tok.note = spec.note
    if spec.src is None:
        tok.status = {'reference': 'structural', 'raw-data': 'raw-data'}.get(spec.cat, 'design')
        tok.source = spec.note
        return
    try:
        s = spec.src() if callable(spec.src) else spec.src
    except Exception as e:     # a missing row is an error, never a silent pass
        tok.status, tok.source = 'error', f'{type(e).__name__}: {e}'
        return
    if not isinstance(s, S):
        s = S(s, spec.note or 'constant')
    tok.source, tok.files = s.desc, s.files
    tok.expected = fmt(s.value * spec.scale, tok.ndec) if spec.op == 'eq' else f'{spec.op} {s.value * spec.scale:.6g}'
    ok = same(tok, s.value, spec.op, spec.scale)
    if spec.cat == 'design':
        tok.status = 'design-ok' if ok else 'mismatch'
    else:
        tok.status = 'ok' if ok else 'mismatch'


class Engine:
    def __init__(self, files):
        self.lines, self.toks = {}, []
        for tag, path in files.items():
            ls, ts = tokenize(path, tag)
            self.lines[tag] = ls
            self.toks += ts
        self.problems = []
        self.marks = []

    def _locate(self, tag, anchor):
        ls = self.lines[tag]
        hits = [(i, j) for i, l in enumerate(ls) for j in [m.start() for m in re.finditer(re.escape(anchor), l)]]
        return hits

    def text(self, tag, anchor, specs, rule=None, every=False):
        """Numbers inside `anchor` (a literal piece of one line), in order, get these specs.
        The anchor must occur exactly once, unless every=True (formula constants)."""
        hits = self._locate(tag, anchor)
        rule = rule or anchor[:60]
        if not hits or (len(hits) != 1 and not every):
            self.problems.append(f'[{tag}] anchor found {len(hits)} times: {anchor!r}')
            return
        for i, j in hits:
            span = [t for t in self.toks if t.file == tag and t.line == i + 1 and j <= t.col and t.end <= j + len(anchor)]
            if len(span) != len(specs):
                self.problems.append(f'[{tag}:{i+1}] {len(span)} numbers but {len(specs)} specs in: {anchor!r} '
                                     f'-> {[t.text for t in span]}')
                continue
            for t, sp in zip(span, specs):
                if t.status != 'unchecked':
                    if every:
                        continue
                    self.problems.append(f'[{tag}:{i+1}] number {t.text} covered twice ({t.rule} / {rule})')
                apply_spec(t, sp, rule)

    def pattern(self, regex, spec, tags=None, rule=None, group=1):
        """Every unchecked number that sits at group `group` of `regex` gets `spec`."""
        rx = re.compile(regex)
        n = 0
        for tag, ls in self.lines.items():
            if tags and tag not in tags:
                continue
            for i, l in enumerate(ls):
                for m in rx.finditer(l):
                    a, b = m.span(group)
                    for t in self.toks:
                        if t.file == tag and t.line == i + 1 and a <= t.col and t.end <= b and t.status == 'unchecked':
                            apply_spec(t, spec, rule or regex)
                            n += 1
        return n

    def table_rows(self, tag, label):
        """(line_no, [cells]) of the body rows of the tabular that follows \\label{label}."""
        ls = self.lines[tag]
        i0 = next(i for i, l in enumerate(ls) if f'\\label{{{label}}}' in l)
        i = next(k for k in range(i0, len(ls)) if '\\midrule' in ls[k])
        rows = []
        for k in range(i + 1, len(ls)):
            l = ls[k]
            if '\\bottomrule' in l:
                break
            if '&' not in l:
                continue
            rows.append((k, l))
        return rows

    def cells(self, tag, k):
        """Tokens of line k grouped by '&' column."""
        l = self.lines[tag][k]
        bounds, pos = [], 0
        for part in l.split('&'):
            bounds.append((pos, pos + len(part)))
            pos += len(part) + 1
        out = []
        for a, b in bounds:
            out.append([t for t in self.toks if t.file == tag and t.line == k + 1 and a <= t.col < b])
        return out, [l[a:b] for a, b in bounds]
# ------------------------------------------------------------- sources
VA = 'T3.10_main_figures/values_all.csv'
TA = 'T3.10_main_figures/tests_all.csv'
CFG = 'T1.5_leakage_audit/causal/config_table.csv'
RT = 'T3.8_runtime/runtime_paper_table.csv'
RS = 'T3.8_runtime/real_summary_all.csv'
AB = 'T3.3_ablation/ablation_summary.csv'
ABT = 'T1.6_stats/T3.3_ablation/all_paired_tests.csv'
GR = 'T3.2_param_grid/grid_summary.csv'
SEL = 'T3.2_param_grid/selection/selection.csv'
RE_ = 'T2.4_responsiveness/responsiveness_summary.csv'
RET = 'T2.4_responsiveness/all_paired_tests.csv'
RF = 'T2.4_responsiveness/all_rsi_failures.csv'
EX = 'T2.4_responsiveness/all_example_events.csv'
RB = 'T3.5_robustness/robustness_summary.csv'
RBT = 'T1.6_stats/T3.5_robustness/all_paired_tests.csv'
SH = 'T3.6_shift_invariance/shift_summary.csv'
SY = 'T3.6_shift_invariance/synthetic/synthetic_summary.csv'
FR = 'T3.7_feature_relation/feature_relation_summary.csv'
AT = 'T3.7_feature_relation/alpha_beta_total.csv'
PD = 'T3.4_padding/padding_summary.csv'
PS = 'T3.4_padding/padded_share.csv'
BW = 'T3.4_padding/boundary_weight.csv'
DS = 'T3.11_youtube_provenance/dataset_summary.csv'
YP = 'T3.11_youtube_provenance/raw_profile.csv'
TR = 'T1.5_leakage_audit/causal/truncation_test.csv'
PT = 'T1.5_leakage_audit/causal/perturbation_test.csv'
LJ = 'T1.5_leakage_audit/causal/metadata/leakage_audit_run.json'
GP = 'T3.9_movielens/data_prep/granularity_profile.csv'
YR = 'T3.9_movielens/data_prep/year_profile.csv'
MP = 'T3.9_movielens/data_prep/prep_summary.json'
SW = 'T2.2_window_sweep/sweep_summary.csv'
RG = 'T3.8_runtime/bench/runtime_grid.csv'
RSS = 'T3.8_runtime/memory/rss_grid.csv'
SM = 'T3.8_runtime/memory/state_memory.csv'
BJ = 'T3.8_runtime/bench/metadata/bench_run.json'

M4 = ['youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min']
ML = ['movielens_daily', 'movielens_weekly']
M6 = M4 + ML
BASE6 = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']
BASE5 = [m for m in BASE6 if m != 'PFRF']
ALL9 = BASE6 + ['DWT+AF', 'DTCWT+AF', 'WSPI']
NDCG, RHO, RSI, DR = 'ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion'


def V(sc, m, met, cfg='default', col='mean'):
    return Q(VA, col, config=cfg, scenario=sc, method=m, metric=met)


def nwin(sc, cfg='default'):
    return V(sc, 'WSPI', NDCG, cfg, 'n_windows')


def ssum(xs, label):
    xs = list(xs)
    return S(sum(x.value for x in xs), f'sum over {len(xs)} {label}: ' + '; '.join(x.desc for x in xs[:1]) + ' ...',
             sum((x.files for x in xs), ()))


def cfgv(method, col, sc='youtube_hourly'):
    return Q(CFG, col, run_group='T1.4_protocol_v5', scenario=sc, method=method)


def ratio_dr(fn):
    """Per scenario: lowest dRank of the five baselines other than PFRF / dRank of WSPI (default)."""
    rs = [smin([V(sc, m, DR) for m in BASE5]) / V(sc, 'WSPI', DR) for sc in M6]
    return fn(rs, ' scenarios (baseline/WSPI dRank)')


def resp(sc, m, col, cfg='default', variant='main'):
    return Q(RE_, col, scenario=sc, config=cfg, variant=variant, method=m)


def rb(sc, cfg, m, cond, col='dr_mean'):
    return Q(RB, col, scenario=sc, config=cfg, method=m, condition=cond)


def grid(sc, a, b, col, part='test'):
    return Q(GR, col, scenario=sc, alpha=float(a), beta=float(b), part=part)


def fr(sc, stat, kind='within_window', col='value'):
    return Q(FR, col, scenario=sc, kind=kind, statistic=stat)


def verdict(cfg, sc, met, m):
    df = load(TA)
    r = df[(df.config == cfg) & (df.scenario == sc) & (df.metric == met) & (df.method == m)]
    if len(r) != 1:
        raise LookupError(f'{TA}: {len(r)} rows for {cfg},{sc},{met},{m}')
    return r.iloc[0]['verdict']


def mark(E, tag, line, what, expected, found, source):
    E.marks.append(dict(file=tag, line=line, item=what, expected=expected, found=found,
                        status='ok' if expected == found else 'mismatch', source=source))


def body_lines(E, tag, label):
    ls = E.lines[tag]
    i0 = next(i for i, l in enumerate(ls) if f'\\label{{{label}}}' in l)
    i = next(k for k in range(i0, len(ls)) if '\\midrule' in ls[k])
    out = []
    for k in range(i + 1, len(ls)):
        if '\\bottomrule' in ls[k]:
            break
        out.append(k)
    return out


# ------------------------------------------------------ table checks
SCEN_LABEL = {'YouTube Hourly': 'youtube_hourly', 'NYC Yellow Taxi Hourly': 'taxi_hourly',
              'NYC Yellow Taxi 30m': 'taxi_30min', 'NYC Yellow Taxi 5m': 'taxi_5min'}


def plain(s):
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\textsuperscript\{[^}]*\}', '', s)
    return s.replace('\\\\', '').strip()


HEAD_SCEN = {'YouTube\\(1h)': 'youtube_hourly', 'Taxi\\Hourly': 'taxi_hourly', 'Taxi\\30min': 'taxi_30min',
             'Taxi\\5min': 'taxi_5min', 'MovieLens\\(1d)': 'movielens_daily', 'MovieLens\\(1w)': 'movielens_weekly'}
PANEL_MET = {'NDCG@10': NDCG, 'Spearman $\\rho$': RHO, 'RSI@10': RSI, '$\\Delta$Rank': DR}


def table_main(E, label, cfg):
    """Tables 8 and 9 (layout of task T4.12): one panel per metric (row '\\textit{<metric> <arrow>}'),
    one column per scenario (column heads \\shortstack{...} as in Table 5)."""
    tag = 'main'
    ls = E.lines[tag]
    i0 = next(i for i, l in enumerate(ls) if f'\\label{{{label}}}' in l)
    head = next(ls[k] for k in range(i0, len(ls)) if ls[k].startswith('Method &'))
    scen = [HEAD_SCEN[h.replace(chr(92) * 2, chr(92))] for h in re.findall(r'\\shortstack\{([^}]*)\}', head)]
    assert scen == M6, (label, scen)
    met = None
    for k in body_lines(E, tag, label):
        l = ls[k]
        m = re.search(r'\\textit\{([^}]*?) \$\\(?:up|down)arrow\$\}', l)
        if '\\multicolumn' in l and m:
            met = PANEL_MET[m.group(1)]
            continue
        if '&' not in l:
            continue
        toks, raw = E.cells(tag, k)
        meth = plain(raw[0])
        for j, sc in enumerate(scen):
            ts = toks[j + 1]
            assert len(ts) == 1, (k, ts)
            apply_spec(ts[0], R(lambda sc=sc, meth=meth, met=met: V(sc, meth, met, cfg)), f'{label} cell')
            cell = raw[j + 1]
            # significance mark against the block Wilcoxon verdict (reference WSPI)
            if meth != 'WSPI':
                v = verdict(cfg, sc, met, meth)
                exp = {'ref_worse': 'up', 'ref_better': 'down', 'n.s.': 'none'}[v]
            else:
                exp = 'none'
            found = 'up' if 'blacktriangle' in cell else ('down' if 'triangledown' in cell else 'none')
            mark(E, tag, k + 1, f'{label} {sc} {meth} {met} mark', exp, found, f'{TA} verdict')
            # bold = best of the nine methods in the scenario
            vals = {mm: V(sc, mm, met, cfg).value for mm in ALL9}
            best = (min if met == DR else max)(vals, key=vals.get)
            rnd = 2 if met == DR else 4
            is_best = round(vals[meth], rnd) == round(vals[best], rnd)
            mark(E, tag, k + 1, f'{label} {sc} {meth} {met} bold', str(is_best), str('\\textbf' in cell), f'{VA} mean, best of nine')


def year_of(sc, fld):
    v = str(load(DS).set_index('dataset').loc[sc, fld])
    return S(int(v[:4]), f'{DS} [dataset={sc}] {fld} (year)', (DS,))


def table_data(E):
    tag = 'main'
    unit = {'hour': 60, 'min': 1, 'day': 1440, 'week': 10080}
    for k in body_lines(E, tag, 'tab:data'):
        toks, raw = E.cells(tag, k)
        name = plain(raw[0])
        for j, sc in enumerate(M6):
            ts = toks[j + 1]
            if name.startswith('Items'):
                apply_spec(ts[0], R(lambda sc=sc: Q(DS, 'items_admitted_causal', dataset=sc)), 'tab:data items')
            elif name == 'Common windows':
                apply_spec(ts[0], R(lambda sc=sc: nwin(sc)), 'tab:data windows')
            elif name == 'Granularity':
                u = unit[plain(raw[j + 1]).split()[1]]
                apply_spec(ts[0], D('slot length', (lambda sc=sc, u=u: Q(DS, 'slot_minutes', dataset=sc) / u)), 'tab:data slot')
            elif name == 'Period':
                for q, t in enumerate(ts):
                    apply_spec(t, D('first or last year of the data', (lambda sc=sc, q=q: year_of(sc, 'first_time' if q == 0 else 'last_time'))), 'tab:data period')
            else:
                for t in ts:
                    apply_spec(t, X('table text'), 'tab:data')


def table_config(E):
    tag = 'main'
    for k in body_lines(E, tag, 'tab:config'):
        toks, raw = E.cells(tag, k)
        meth = plain(raw[0])
        apply_spec(toks[1][0], D('window length N, default configuration', (lambda meth=meth: cfgv(meth, 'window_slots'))), 'tab:config N')
        apply_spec(toks[2][0], D('observed slots required in the window', (lambda meth=meth: cfgv(meth, 'min_obs'))), 'tab:config min slots')
        line = E.lines[tag][k]
        for t in toks[3]:
            if t.value == 3 and line[max(0, t.col - 2):t.col] == 'J=':
                apply_spec(t, D('decomposition level J', (lambda meth=meth: cfgv(meth, 'level_J'))), 'tab:config J')
            elif t.value == 1 and 'beta=' in line[max(0, t.col - 6):t.col]:
                apply_spec(t, D('default weights alpha=beta=1'), 'tab:config weights')
            else:
                apply_spec(t, D('fixed parameter of the scorer (evaluation/method_configs.py; not tuned)'), 'tab:config parameter')


def table_libs(E, versions):
    tag = 'main'
    for k in body_lines(E, tag, 'tab:libs'):
        toks, raw = E.cells(tag, k)
        name = plain(raw[0]).split(' ')[0]
        ver = plain(raw[1])
        exp = versions.get(name.lower())
        for t in toks[1]:
            t.rule, t.category = 'tab:libs version', 'result'
            t.expected, t.source = str(exp), 'version recorded in the run metadata (see --versions)'
            t.status = 'ok' if exp == ver else 'mismatch'


def table_runtime(E):
    tag = 'main'
    cols = ['default_N', 'us_per_item_default_M1e4', 's_all_items_default_M1e6', 'ms_single_item_call_default',
            'traced_bytes_per_item_default_B1e4', 'latency_ms_median_youtube_hourly', 'latency_ms_median_taxi_5min']
    for k in body_lines(E, tag, 'tab:runtime'):
        toks, raw = E.cells(tag, k)
        meth = plain(raw[0])
        for j, c in enumerate(cols):
            apply_spec(toks[j + 1][0], (D if j == 0 else R)(*(['window length', (lambda meth=meth, c=c: Q(RT, c, method=meth))] if j == 0 else [(lambda meth=meth, c=c: Q(RT, c, method=meth))])), 'tab:runtime')


ABL_ROWS = {'WSPI (full, DTCWT)': 'WSPI', 'Trend only ($\\mu_L$)': 'Trend', 'Trend $+\\,R$ ($\\mu_L e^{R}$)': 'Trend+R',
            'Trend $+\\,W_E$ ($\\mu_L e^{-W_E}$)': 'Trend+WE', 'WSPI with DWT': 'DWT-WSPI', 'Trend only with DWT': 'DWT-Trend'}


def table_ablation(E):
    tag = 'main'
    mets = [NDCG, RSI, DR]
    for k in body_lines(E, tag, 'tab:ablation'):
        l = E.lines[tag][k]
        if '&' not in l or 'Variant' in l:
            continue
        toks, raw = E.cells(tag, k)
        var = ABL_ROWS[raw[0].strip()]
        for t in toks[0]:
            apply_spec(t, X('formula in the row label'), 'tab:ablation label')
        for j in range(6):
            sc = 'youtube_hourly' if j < 3 else 'taxi_hourly'
            met = mets[j % 3]
            apply_spec(toks[j + 1][0], R(lambda sc=sc, var=var, met=met: Q(AB, f'{met}_mean', scenario=sc, family='ablation', variant=var)), 'tab:ablation cell')
            if var != 'WSPI':
                vv = load(ABT)
                vv = vv[(vv.run_group == sc) & (vv.scenario == 'ablation') & (vv.method == var) & (vv.metric == met)].iloc[0]['verdict']
                mark(E, tag, k + 1, f'tab:ablation {sc} {var} {met} dagger', str(vv == 'n.s.'), str('dagger' in raw[j + 1]), f'{ABT} verdict')


def table_sens(E):
    tag = 'main'
    cols = ['ndcg@10_mean', 'rsi@10_mean', 'robustness_distortion_mean']
    for k in body_lines(E, tag, 'tab:sens'):
        l = E.lines[tag][k]
        if '&' not in l:
            continue
        toks, raw = E.cells(tag, k)
        lab = raw[0]
        if 'alpha=\\beta=1' in lab:
            a = b = 1.0
        else:
            m1 = re.search(r'\\(alpha|beta)=([0-9.]+)', lab)
            a, b = (float(m1.group(2)), 1.0) if m1.group(1) == 'alpha' else (1.0, float(m1.group(2)))
        for t in toks[0]:
            apply_spec(t, D('grid value of alpha or beta (Section 4.9)'), 'tab:sens label')
        for j in range(6):
            sc = 'youtube_hourly' if j < 3 else 'taxi_hourly'
            apply_spec(toks[j + 1][0], R(lambda sc=sc, a=a, b=b, c=cols[j % 3]: grid(sc, a, b, c)), 'tab:sens cell')


def table_resp(E):
    tag = 'main'
    # header: methods and N
    cols = [('WSPI', 'default'), ('DTCWT+AF', 'default'), ('DWT+AF', 'default'), ('AF', 'default'),
            ('RRD', 'default'), ('RRD', 'equal64'), ('CompoundPop', 'equal64')]
    scen = {'YouTube (1h)': 'youtube_hourly', 'Taxi Hourly': 'taxi_hourly', 'Taxi 30min': 'taxi_30min', 'Taxi 5min': 'taxi_5min'}
    for k in body_lines(E, tag, 'tab:resp'):
        toks, raw = E.cells(tag, k)
        sc = scen[plain(raw[0])]
        for t in toks[0]:
            apply_spec(t, X('row label'), 'tab:resp label')
        for j, (m, cfg) in enumerate(cols):
            ts = toks[j + 1]
            apply_spec(ts[0], R(lambda sc=sc, m=m, cfg=cfg: resp(sc, m, 'miss_rate', cfg)), 'tab:resp miss rate')
            undefined = str(load(RE_).query('scenario==@sc and config==@cfg and variant=="main" and method==@m').iloc[0]['median_undefined']) == 'True'
            if len(ts) == 2:
                apply_spec(ts[1], R(lambda sc=sc, m=m, cfg=cfg: resp(sc, m, 'median_delay', cfg)), 'tab:resp median delay')
            mark(E, tag, k + 1, f'tab:resp {sc} {m} {cfg} median shown', str(not undefined), str(len(ts) == 2), f'{RE_} median_undefined')
    # the header line with (64)/(7)
    E.pattern(r'\\\\\((64)\)\}', D('window length N=64', lambda: cfgv('WSPI', 'window_slots')), tags=['main'], rule='tab:resp header')
    E.pattern(r'\\\\\((7)\)\}', D('window length N=7', lambda: cfgv('AF', 'window_slots')), tags=['main'], rule='tab:resp header')


def table_compare(E):
    tag = 'main'
    for k in body_lines(E, tag, 'tab:compare'):
        toks, raw = E.cells(tag, k)
        for t in toks[0]:
            apply_spec(t, X('row number of Table 1'), 'tab:compare')
        # citations and complexity formulae are handled by the generic handlers


def table_strengths(E, sections):
    tag = 'main'
    for k in body_lines(E, tag, 'tab:strengths'):
        toks, raw = E.cells(tag, k)
        for t in toks[-1]:
            ok = t.text in sections
            t.rule, t.category, t.source = 'tab:strengths section column', 'reference', 'section number in the main aux'
            t.status = 'structural' if ok else 'mismatch'
            t.expected = 'existing section' if ok else 'no such section'
# ------------------------------------------------ text rules, main paper
def pad_maxdiff(cols):
    df = load(PD)
    d = df[(df.layer == 'pad') & (df.method == 'WSPI') & (df.subset == 'all')]
    vals = []
    for sc in M4:
        ref = d[(d.scenario == sc) & (d['mode'] == 'reflect')].iloc[0]
        for _, r in d[(d.scenario == sc) & (d['mode'] != 'reflect')].iterrows():
            vals += [abs(r[c] - ref[c]) for c in cols]
    return S(max(vals), f'{PD} [layer=pad, method=WSPI, subset=all] max |mode - reflect| of {cols}, 4 scenarios x 4 modes', (PD,))


def yp(key):
    return Q(YP, 'value', key=key)


def ytime(key, part):
    ts = pd.Timestamp(str(load(YP).set_index('key').loc[key, 'value']).split(';')[0])
    return S(getattr(ts, part), f'{YP} [key={key}] {part}', (YP,))


def ab(sc, var, met, fam='ablation'):
    return Q(AB, f'{met}_mean', scenario=sc, family=fam, variant=var)


def fusion_spread(met):
    vals = []
    for sc in M4:
        v = [ab(sc, x, met, 'fusion').value for x in ['WSPI', 'Linear', 'Product', 'Additive']]
        vals.append(max(v) - min(v))
    return S(max(vals), f'{AB} [family=fusion] max over 4 scenarios of (max - min over 4 fusion forms) {met}_mean', (AB,))


def fusion_gap(met):
    vals = []
    for sc in M4:
        w = ab(sc, 'WSPI', met, 'fusion').value
        vals += [abs(ab(sc, x, met, 'fusion').value - w) for x in ['Linear', 'Product', 'Additive']]
    return S(max(vals), f'{AB} [family=fusion] max over 4 scenarios x 3 alternatives of |alternative - exponential (WSPI)| {met}_mean', (AB,))


def grid_spread(col):
    d = load(GR)
    d = d[d.part == 'test']
    v = d.groupby('scenario')[col].agg(lambda x: x.max() - x.min())
    return S(v.max(), f'{GR} [part=test] max over scenarios of (max - min over 49 settings) {col}', (GR,))


def grid_total(sc, total, fn):
    df = load(GR)
    d = df[(df.part == 'test') & (df.scenario == sc) & ((df.alpha + df.beta - total).abs() < 1e-9)]
    return S(fn(d.robustness_distortion_mean), f'{GR} [part=test, scenario={sc}, alpha+beta={total}] {fn.__name__} robustness_distortion_mean ({len(d)} splits)', (GR,))


def sel(sc, param, col):
    return Q(SEL, col, scenario=sc, param=param)


def RET_(sc, m, cfg='default', variant='main', col='diff_ref_minus_method'):
    return Q(RET, col, scenario=sc, config=cfg, variant=variant, reference='WSPI', method=m)


def slot(sc):
    return lambda: Q(DS, 'slot_minutes', dataset=sc)


def mp_date(key, part):
    s = load(MP)['rules'][key]
    v = {'day': int(s[8:10]), 'year': int(s[:4])}[part]
    return S(v, f'{MP} rules/{key} ({part})', (MP,))


def rules_main_text_1(E):
    t = lambda anchor, *specs, **kw: E.text('main', anchor, list(specs), **kw)
    sum6 = lambda: ssum([nwin(sc) for sc in M6], 'scenarios, common windows of WSPI (NDCG@10)')
    rmin, rmax = (lambda: ratio_dr(smin)), (lambda: ratio_dr(smax))
    t('over 131,443 windows, with block-bootstrap confidence intervals. With', R(sum6))
    t('2.0 to 2.8 times lower rank displacement', R(rmin), R(rmax))
    t('over 131,443 evaluation windows', R(sum6))
    t('over 131,443 windows, with block-bootstrap confidence intervals and paired tests', R(sum6))
    t('about a 56\\% improvement', X('number quoted from the cited study [24]'))
    t('98.33\\% accuracy', X('number quoted from the cited study [40]'))
    t('(one hour, 30 minutes, 5 minutes,', D('slot length (min)', slot('taxi_30min')), D('slot length (min)', slot('taxi_5min')))
    t('(0 if the slot has no record)', D('a slot without a record counts as zero (protocol V5)'))
    t('(64 for the wavelet-based methods, 7 for the baselines by default)', D('window length', lambda: cfgv('WSPI', 'window_slots')), D('window length', lambda: cfgv('AF', 'window_slots')))
    t('(1 slot)', D('evaluation horizon T_future = 1 (protocol V5)'))
    t('(16 for $N=64$, $J=3$)', D('n_L = N/2^(J-1)', S(64 / 2 ** 2, '64 / 2^(3-1)')), D('window length', lambda: cfgv('WSPI', 'window_slots')), D('level', lambda: cfgv('WSPI', 'level_J')))
    t('redundant by only 2:1', X('redundancy of DTCWT, property of the transform [27]'), X('redundancy of DTCWT'))
    t('(windows 32 to 63)', D('first padded window', lambda: Q(PS, 'first_padded', scenario='youtube_hourly', method='WSPI')), D('last padded window', lambda: Q(PS, 'last_padded', scenario='youtube_hourly', method='WSPI')))
    t('This concerns 32 windows per scenario, from 4.9\\% (YouTube) to 0.03\\% (5-minute taxi)',
      R(lambda: smax([Q(PS, 'padded_windows', scenario=sc, method='WSPI') for sc in M4])),
      P(lambda: Q(PS, 'padded_share', scenario='youtube_hourly', method='WSPI')),
      P(lambda: Q(PS, 'padded_share', scenario='taxi_5min', method='WSPI')), D('slot length (min)', slot('taxi_5min')))
    t('about 30\\% of the filter weight', P(lambda: Q(BW, 'extended_weight_share', coefficient=16)))
    t('by at most 0.002 and its rank distortion by at most 0.2', R(lambda: pad_maxdiff(['ndcg@10_mean', 'rsi@10_mean'])), R(lambda: pad_maxdiff(['robustness_distortion_mean'])))
    t('on the 5-minute taxi data edge extension was 0.0005 higher', D('slot length (min)', slot('taxi_5min')),
      R(lambda: Q(PD, 'ndcg@10_mean', scenario='taxi_5min', layer='ext', method='WSPI', mode='edge', subset='all') - Q(PD, 'ndcg@10_mean', scenario='taxi_5min', layer='ext', method='WSPI', mode='symmetric', subset='all')))
    t('near $1$ indicates', X('value of R near 1 (description)'))
    t('their Spearman correlation is $-0.993$ to $-0.998$', R(lambda: smax([fr(sc, 'sp_R_WE') for sc in M4])), R(lambda: smin([fr(sc, 'sp_R_WE') for sc in M4])))
    t('roughly $[0.37,2.72]$', D('e^-1 (alpha=beta=1)', S(math.exp(-1), 'exp(-1)')), D('e^1', S(math.exp(1), 'exp(1)')))
    t('$\\alpha,\\beta\\in\\{0,0.25,0.5,0.75,1,1.5,2\\}$', *[D('grid value (T3.2)', (lambda k=k: S(sorted(load(GR).alpha.unique())[k], f'{GR} sorted unique alpha [{k}]', (GR,)))) for k in range(7)])
    t('first 30\\% of the evaluation windows and tested on the remaining 70\\%',
      P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_tune') / sel('youtube_hourly', 'alpha_beta', 'n_common')),
      P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_test') / sel('youtube_hourly', 'alpha_beta', 'n_common')))
    t('this leaves 16 approximation coefficients', D('n_L = N/2^(J-1)', S(16, '64 / 2^(3-1)')))
    t('(10,000 resamples)', D('bootstrap resamples (tools/stats_report.py, T1.6)'))
    t('one day on YouTube (24 slots)', D('bootstrap block', lambda: V('youtube_hourly', 'WSPI', NDCG, col='block')))
    t('(168, 336 and 2,016 slots)', *[D('bootstrap block', (lambda sc=sc: V(sc, 'WSPI', NDCG, col='block'))) for sc in M4[1:]])
    t('the block is 7 days for daily data and 4 weeks for weekly data', D('bootstrap block', lambda: V('movielens_daily', 'WSPI', NDCG, col='block')), D('bootstrap block', lambda: V('movielens_weekly', 'WSPI', NDCG, col='block')))
    t('(ML-32M) [43] adds', X('dataset name ML-32M'), X('citation'))
    t('ML-32M is a public', X('dataset name ML-32M'))
    t('(CC0, version 2) [44]', D('dataset version on Kaggle'), X('citation'))
    t('about 1,500 videos uploaded in April 2018 on 7 May 2018', X('from the Kaggle dataset description (no CSV)'), X('from the Kaggle dataset description (no CSV)'),
      D('first snapshot, day', lambda: ytime('raw_first_time', 'day')), D('first snapshot, year', lambda: ytime('raw_first_time', 'year')))
    t('The file covers 1,611 videos and 695 hourly snapshots, from 7 May 2018 18:00 to 5 June 2018 16:00',
      R(lambda: yp('raw_videos')), R(lambda: yp('raw_hours')),
      D('first snapshot, day', lambda: ytime('raw_first_time', 'day')), D('year', lambda: ytime('raw_first_time', 'year')),
      D('hour', lambda: ytime('raw_first_time', 'hour')), D('minute', lambda: ytime('raw_first_time', 'minute')),
      D('last snapshot, day', lambda: ytime('raw_last_time', 'day')), D('year', lambda: ytime('raw_last_time', 'year')),
      D('hour', lambda: ytime('raw_last_time', 'hour')), D('minute', lambda: ytime('raw_last_time', 'minute')))
    t('a collection gap on 19 May 2018', D('gap day', lambda: ytime('gap_hours_list', 'day')), D('gap year', lambda: ytime('gap_hours_list', 'year')))
    t('(4,356 rows, 0.4\\%)', R(lambda: yp('diff_negative_rows')), P(lambda: yp('diff_negative_share_of_rows')))
    t('at least 50 new views over the period', D('video filter of the converter', lambda: yp('processed_min_video_total')))
    t('This leaves 1,485 videos and 999,155 video-hours', R(lambda: yp('processed_videos')), R(lambda: yp('processed_rows')))
    t('inside the input of 64 of the 649 common windows', R(lambda: yp('common_windows_input64_contains_gap')), R(lambda: yp('common_windows')))
    t('by more than 0.0022', R(lambda: yp('gap_sensitivity_max_abs_diff')))
    t('The 50-view filter', D('video filter of the converter', lambda: yp('processed_min_video_total')))
    t('threshold: 50 views on YouTube and 24 trips or ratings', D('item threshold', lambda: Q(DS, 'min_obs', dataset='youtube_hourly')), D('item threshold', lambda: Q(DS, 'min_obs', dataset='taxi_hourly')))
    t('first 25\\% of the data', D('truncation fraction of the audit', lambda: J(LJ, 'truncate_frac') * 100))
    t('(largest relative difference $3.2\\times10^{-16}$)', R(lambda: S(load(TR).max_rel_diff.max(), f'{TR} max max_rel_diff ({len(load(TR))} runs)', (TR,))))
    t('the 63 slots after it', D('perturbed slots of the audit minus the test slot', lambda: J(LJ, 'perturb_slots') - 1))
    t('the $\\Delta$Rank test uses 50 items and seed 42', D('number of target items (protocol V5)'), D('seed', lambda: cfgv('WSPI', 'seed')))
    t('$N\\in\\{7,16,32,64,128\\}$, with a simple', *[D('window length of the sweep', (lambda k=k: S(sorted(load(SW).window.astype(int).unique())[k], f'{SW} sorted unique window [{k}]', (SW,)))) for k in range(5)])
def rules_main_text_2(E):
    t = lambda anchor, *specs, **kw: E.text('main', anchor, list(specs), **kw)
    t('(0.9493 to 0.9897, against 0.8619 to 0.9446 for WSPI)',
      R(lambda: smin([V(sc, 'RRD', RSI, 'equal64') for sc in M6])), R(lambda: smax([V(sc, 'RRD', RSI, 'equal64') for sc in M6])),
      R(lambda: smin([V(sc, 'WSPI', RSI, 'equal64') for sc in M6])), R(lambda: smax([V(sc, 'WSPI', RSI, 'equal64') for sc in M6])))
    # Tables 8 and 9, six scenarios (T4.12)
    for cfg, a in [('default', 'YouTube Hourly 649, NYC Yellow Taxi Hourly 7,983, NYC Yellow Taxi 30m 15,998, NYC Yellow Taxi 5m 96,145, MovieLens Daily 9,356, MovieLens Weekly 1,312'),
                   ('equal64', 'YouTube Hourly 652, NYC Yellow Taxi Hourly 7,983, NYC Yellow Taxi 30m 15,998, NYC Yellow Taxi 5m 96,148, MovieLens Daily 9,357, MovieLens Weekly 1,312')]:
        t(a, R(lambda cfg=cfg: nwin('youtube_hourly', cfg)), R(lambda cfg=cfg: nwin('taxi_hourly', cfg)), D('slot', slot('taxi_30min')), R(lambda cfg=cfg: nwin('taxi_30min', cfg)),
          D('slot', slot('taxi_5min')), R(lambda cfg=cfg: nwin('taxi_5min', cfg)), R(lambda cfg=cfg: nwin('movielens_daily', cfg)), R(lambda cfg=cfg: nwin('movielens_weekly', cfg)))
    for cfg, anchor, meths in [('default', 'PFRF (1.00), VSE (0.99), RRD (0.99), CompoundPop (0.74), AF (0.43)', ['PFRF', 'VSE', 'RRD', 'CompoundPop', 'AF']),
                               ('equal64', 'PFRF (1.00), VSE (0.63), RRD (0.57), CompoundPop (0.12)', ['PFRF', 'VSE', 'RRD', 'CompoundPop'])]:
        t(anchor, *[R(lambda m=m, cfg=cfg: smax([V(sc, m, NDCG, cfg, 'ties_top21_share') for sc in M6])) for m in meths])
        df = load(VA)
        d = df[(df.config == cfg) & (df.scenario.isin(M6)) & (df.metric == NDCG)]
        over = sorted(d[d.method.isin(ALL9)].groupby('method').ties_top21_share.max().loc[lambda x: x >= 0.10].index)
        E.marks.append(dict(file='main', line=0, item=f'tie footnote {cfg}: methods with share >= 0.10', expected=', '.join(over),
                            found=', '.join(sorted(meths)), status='ok' if over == sorted(meths) else 'mismatch', source=f'{VA} ties_top21_share'))
    t('in 10\\% or more of the windows', D('threshold of the tie footnote (10%)'), every=True)
    t('AF has the highest value (0.9842), and WSPI reaches 0.9716', R(lambda: V('youtube_hourly', 'AF', NDCG)), R(lambda: V('youtube_hourly', 'WSPI', NDCG)))
    gap = lambda sc, m: V(sc, m, NDCG) - V(sc, 'WSPI', NDCG)
    t('The gap to the best method is 0.058 at the hourly level (AF, 0.9403), 0.056 at 30 minutes (AF, 0.9549) and 0.004 at 5 minutes (EWMA, 0.9102)',
      R(lambda: gap('taxi_hourly', 'AF')), R(lambda: V('taxi_hourly', 'AF', NDCG)), R(lambda: gap('taxi_30min', 'AF')), D('slot', slot('taxi_30min')),
      R(lambda: V('taxi_30min', 'AF', NDCG)), R(lambda: gap('taxi_5min', 'EWMA')), D('slot', slot('taxi_5min')), R(lambda: V('taxi_5min', 'EWMA', NDCG)))
    t('AF has the highest $\\rho$ (0.9806) and WSPI reaches 0.9775', R(lambda: V('youtube_hourly', 'AF', RHO)), R(lambda: V('youtube_hourly', 'WSPI', RHO)))
    t('(0.8734 and 0.8657) and DTCWT+AF at 5 minutes (0.8029)', R(lambda: V('taxi_hourly', 'DWT+AF', RHO)), R(lambda: V('taxi_30min', 'DWT+AF', RHO)), D('slot', slot('taxi_5min')), R(lambda: V('taxi_5min', 'DTCWT+AF', RHO)))
    t('from 0.8652 (hourly) to 0.8577 (30 minutes) and 0.7971 (5 minutes)', R(lambda: V('taxi_hourly', 'WSPI', RHO)), R(lambda: V('taxi_30min', 'WSPI', RHO)), D('slot', slot('taxi_30min')),
      R(lambda: V('taxi_5min', 'WSPI', RHO)), D('slot', slot('taxi_5min')))
    t('WSPI has the highest value on YouTube (0.9456), on the 5-minute taxi data (0.9072) and on daily MovieLens data (0.8619)', R(lambda: V('youtube_hourly', 'WSPI', RSI)), D('slot', slot('taxi_5min')), R(lambda: V('taxi_5min', 'WSPI', RSI)), R(lambda: V('movielens_daily', 'WSPI', RSI)))
    t('DTCWT+AF is higher (0.9247 against 0.9051)', R(lambda: V('movielens_weekly', 'DTCWT+AF', RSI)), R(lambda: V('movielens_weekly', 'WSPI', RSI)))
    t('(0.8909 against 0.8888, and 0.8962 against 0.8956)', R(lambda: V('taxi_hourly', 'DTCWT+AF', RSI)), R(lambda: V('taxi_hourly', 'WSPI', RSI)), R(lambda: V('taxi_30min', 'DTCWT+AF', RSI)), R(lambda: V('taxi_30min', 'WSPI', RSI)))
    t('except PFRF on YouTube (0.9440)', R(lambda: V('youtube_hourly', 'PFRF', RSI)))
    t('RRD, with 0.9144 on YouTube, 0.8180, 0.8576 and 0.8545 on the three taxi granularities and 0.7803 on daily MovieLens data; on weekly MovieLens data it is VSE (0.8675)',
      *[R(lambda sc=sc: V(sc, 'RRD', RSI)) for sc in M4 + ['movielens_daily']], R(lambda: V('movielens_weekly', 'VSE', RSI)))
    t('37.12 on YouTube, 20.40, 12.60 and 7.03 on the hourly, 30-minute and 5-minute taxi data, and 276.03 and 560.48 on daily and weekly MovieLens data',
      *[R(lambda sc=sc: V(sc, 'WSPI', DR)) for sc in M4], D('slot', slot('taxi_30min')), D('slot', slot('taxi_5min')), *[R(lambda sc=sc: V(sc, 'WSPI', DR)) for sc in ML])
    t('with 96.02, 35.98, 26.83, 15.61, 435.23 and 1032.16', *[R(lambda sc=sc: V(sc, 'DTCWT+AF', DR)) for sc in M6])
    t('(NDCG@10 = 0.1389 on YouTube and 0.4576 on the hourly taxi data)', D('metric cut-off K=10'), R(lambda: V('youtube_hourly', 'PFRF', NDCG)), R(lambda: V('taxi_hourly', 'PFRF', NDCG)))
    t('raised $\\log\\mu_L$ by 1.03 on average and lowered $\\alpha R-\\beta W_E$ by 0.42', R(lambda: rb('youtube_hourly', 'default', 'WSPI', 'size10', 'dlogmu_mean')), R(lambda: -rb('youtube_hourly', 'default', 'WSPI', 'size10', 'dexpo_mean')))
    t('with spikes of 2 to 50 times the mean', D('smallest spike size (condition size2)'), D('largest spike size (condition size50)'))
    t('lasting 1, 3 or 6 slots', D('spike duration (size10)'), D('duration (dur3)'), D('duration (dur6)'))
    pm = lambda sc, fn: fn([rb(sc, 'default', 'WSPI', c) for c in ['pos_middle', 'pos_first']])
    t('by 41 to 45 ranks on YouTube and 9 to 11 on the hourly taxi data', R(lambda: pm('youtube_hourly', smin)), R(lambda: pm('youtube_hourly', smax)), R(lambda: pm('taxi_hourly', smin)), R(lambda: pm('taxi_hourly', smax)))
    t('as a 24-hour moving average', D('smoothing span of Figure 6'))
    t('for example around 19 May, WSPI falls less', X('date read off Figure 6 (visual reading, no CSV)'))
    t('such as around 29 May, WSPI drops', X('date read off Figure 6 (visual reading, no CSV)'))
    t('over the 7,983 hourly NYC Yellow Taxi windows, as a one-week (168-hour)', R(lambda: nwin('taxi_hourly')), D('smoothing span of Figure 7 (one week)'))
    t('their means differ by 0.002 (0.8888 and 0.8909)', R(lambda: V('taxi_hourly', 'DTCWT+AF', RSI) - V('taxi_hourly', 'WSPI', RSI)), R(lambda: V('taxi_hourly', 'WSPI', RSI)), R(lambda: V('taxi_hourly', 'DTCWT+AF', RSI)))
    t('comes close (0.8807)', R(lambda: V('taxi_hourly', 'PFRF', RSI)))
    t('(0.7587, 0.8180 and 0.8051)', R(lambda: V('taxi_hourly', 'AF', RSI)), R(lambda: V('taxi_hourly', 'RRD', RSI)), R(lambda: V('taxi_hourly', 'DWT+AF', RSI)))
    t('the RSI@10 of WSPI rises (0.8888, 0.8956, 0.9072) and its $\\Delta$Rank falls (20.40, 12.60, 7.03)', D('metric cut-off K=10'), *[R(lambda sc=sc: V(sc, 'WSPI', RSI)) for sc in M4[1:]], *[R(lambda sc=sc: V(sc, 'WSPI', DR)) for sc in M4[1:]])
    t('(AF 0.9403 and 0.9549, EWMA 0.9102, against 0.8824, 0.8986 and 0.9061)', R(lambda: V('taxi_hourly', 'AF', NDCG)), R(lambda: V('taxi_30min', 'AF', NDCG)), R(lambda: V('taxi_5min', 'EWMA', NDCG)),
      *[R(lambda sc=sc: V(sc, 'WSPI', NDCG)) for sc in M4[1:]])
    t('although the gap shrinks to 0.004 at 5 minutes', R(lambda: gap('taxi_5min', 'EWMA')), D('slot', slot('taxi_5min')))
    t('covers 64 hours at the hourly level', D('64 slots of one hour'))
def rules_main_text_3(E):
    t = lambda anchor, *specs, **kw: E.text('main', anchor, list(specs), **kw)
    rmin, rmax = (lambda: ratio_dr(smin)), (lambda: ratio_dr(smax))
    t('for $N=64$ and $J=3$, 16 real low-pass and 56 complex detail values, or 1~KiB',
      D('window length', lambda: cfgv('WSPI', 'window_slots')), D('level', lambda: cfgv('WSPI', 'level_J')),
      D('n_L = N/2^(J-1)', S(16, '64/4')), D('detail coefficients 32+16+8', S(56, '32+16+8')),
      R(lambda: Q(RT, 'coef_bytes_per_item_N64', method='WSPI') / 1024))
    t('(7 for the baselines and 64 for the wavelet-based methods in this study)', D('window length', lambda: cfgv('AF', 'window_slots')), D('window length', lambda: cfgv('WSPI', 'window_slots')))
    t('Intel Core Ultra 7 255H', X('processor model (Supplementary Table S12)'), X('processor model'))
    t('Times are medians of 10 repeats', D('repeats of the benchmark', lambda: S(load(RG).repeats.min(), f'{RG} min repeats', (RG,))))
    t('WSPI needed 11.7~\\textmu s per item and window, against 0.02--0.11~\\textmu s for the baselines and 0.66~\\textmu s for DWT+AF. One million items took 12.5~s.',
      R(lambda: Q(RT, 'us_per_item_default_M1e4', method='WSPI')),
      R(lambda: smin([Q(RT, 'us_per_item_default_M1e4', method=m) for m in BASE6])), R(lambda: smax([Q(RT, 'us_per_item_default_M1e4', method=m) for m in BASE6])),
      R(lambda: Q(RT, 'us_per_item_default_M1e4', method='DWT+AF')), R(lambda: Q(RT, 's_all_items_default_M1e6', method='WSPI')))
    t('doubling of $N$ from 64 to 256, and grew 2.7 times from $N=32$ to $N=64$',
      D('N of the benchmark'), D('largest N of the benchmark', lambda: S(load(RG).N.max(), f'{RG} max N', (RG,))),
      R(lambda: Q(RG, 'us_per_item_median', method='WSPI', N=64, M=10000) / Q(RG, 'us_per_item_median', method='WSPI', N=32, M=10000)),
      D('N of the benchmark'), D('N of the benchmark'))
    t('A call for a single item cost about 1.2~ms', R(lambda: Q(RT, 'ms_single_item_call_default', method='WSPI')))
    t('about 3.2~kB per item', R(lambda: Q(RT, 'traced_bytes_per_item_default_B1e4', method='WSPI') / 1000))
    t('in batches of $10^4$ with $N=64$, the resident memory of the process grew by 39~MiB',
      D('batch size of the RSS run', lambda: Q(RSS, 'chunk', method='WSPI', N=64, M=1000000, case='chunked')), D('window length'), R(lambda: Q(RT, 'rss_delta_MiB_N64_M1e6_chunked', method='WSPI')))
    t('a median of 4.1~ms on YouTube and 1.4~ms on the 5-minute taxi data (at most 130~ms)',
      R(lambda: Q(RT, 'latency_ms_median_youtube_hourly', method='WSPI')), R(lambda: Q(RT, 'latency_ms_median_taxi_5min', method='WSPI')),
      D('slot', slot('taxi_5min')), R(lambda: smax([Q(RT, f'latency_ms_max_{sc}', method='WSPI') for sc in M4])))
    t('scoring took 34\\% of the run', P(lambda: Q(RS, 'score_share', scenario='taxi_5min', method='WSPI')))
    t('Columns 3--6: synthetic counts, items scored in batches of $10^4$; memory = peak traced working memory per item in a batch of $10^4$. Columns 7--8',
      X('column numbers'), X('column numbers'), D('batch size', lambda: Q(RG, 'chunk', method='WSPI', N=64, M=10000)), D('batch size', lambda: Q(RG, 'chunk', method='WSPI', N=64, M=10000)), X('column numbers'), X('column numbers'))
    t('rises from 37.41 to 92.35 on YouTube and from 20.40 to 34.70 on the taxi data',
      R(lambda: ab('youtube_hourly', 'WSPI', DR)), R(lambda: ab('youtube_hourly', 'Trend', DR)), R(lambda: ab('taxi_hourly', 'WSPI', DR)), R(lambda: ab('taxi_hourly', 'Trend', DR)))
    t('(by at most 0.0076). They lower NDCG@10 by at most 0.0042',
      R(lambda: smax([ab(sc, 'WSPI', RSI) - ab(sc, 'Trend', RSI) for sc in M4])), D('metric cut-off K=10'),
      R(lambda: smax([ab(sc, 'Trend', NDCG) - ab(sc, 'WSPI', NDCG) for sc in M4])))
    t('RSI@10 falls from 0.9446 to 0.8906 on YouTube and from 0.8888 to 0.7871 on the taxi data', D('metric cut-off K=10'),
      R(lambda: ab('youtube_hourly', 'WSPI', RSI)), R(lambda: ab('youtube_hourly', 'DWT-WSPI', RSI)), R(lambda: ab('taxi_hourly', 'WSPI', RSI)), R(lambda: ab('taxi_hourly', 'DWT-WSPI', RSI)))
    t('(0.9814 against 0.9715 on YouTube); on the 5-minute data', R(lambda: ab('youtube_hourly', 'DWT-WSPI', NDCG)), R(lambda: ab('youtube_hourly', 'WSPI', NDCG)), D('slot', slot('taxi_5min')))
    t('circularly by 0 to 7 slots', D('smallest shift'), D('largest shift (T3.6 design)'))
    cv = lambda tr, fn: fn([Q(SH, 'mean', scenario=sc, transform=tr, metric='cv_E_3') for sc in M4])
    t('the energy of detail level 3 varies by 5--9\\% across the shifts', D('detail level index'), P(lambda: cv('DTCWT', smin)), P(lambda: cv('DTCWT', smax)))
    t('with DWT it varies by 32--38\\%', P(lambda: cv('DWT', smin)), P(lambda: cv('DWT', smax)))
    rr = lambda fn: fn([Q(SH, 'ratio_dwt_over_dtcwt', scenario=sc, transform='DWT', metric=m) for sc in M4 for m in ['sd_R', 'sd_WE']])
    t('The variation of $R$ and $W_E$ is 7 to 12 times smaller with DTCWT', R(lambda: rr(smin)), R(lambda: rr(smax)))
    sw = lambda name, fn: fn([Q(SY, 'mean', shape=sh, age='recent', kind='score', name=name, metric='share_wrong') for sh in ['spike', 'burst3']])
    t('falls in 7--25\\% of the steps', P(lambda: sw('WSPI', smin)), P(lambda: sw('WSPI', smax)))
    t('with DWT features it falls in 56--66\\%', P(lambda: sw('DWT-WSPI', smin)), P(lambda: sw('DWT-WSPI', smax)))
    t('DWT+AF falls in 65--75\\% of the steps', P(lambda: sw('DWT+AF', smin)), P(lambda: sw('DWT+AF', smax)))
    t('The RSI@10 of each alternative differs from that of the exponential form by less than 0.007', D('metric cut-off K=10'), R(lambda: fusion_gap(RSI), op='lt'))
    t('its NDCG@10 differs by less than 0.005', D('metric cut-off K=10'), R(lambda: fusion_gap(NDCG), op='lt'))
    ntune = lambda: sel('youtube_hourly', 'alpha_beta', 'n_tune') / sel('youtube_hourly', 'alpha_beta', 'n_common')
    ntest = lambda: sel('youtube_hourly', 'alpha_beta', 'n_test') / sel('youtube_hourly', 'alpha_beta', 'n_common')
    t('the first 30\\% are used for tuning and the last 70\\% for testing', P(ntune), P(ntest))
    t('within 1\\% of the best one', D('admissible if NDCG@10 >= 0.99 x best', lambda: (1 - sel('youtube_hourly', 'alpha_beta', 'ndcg_threshold') / sel('youtube_hourly', 'alpha_beta', 'ndcg_star')) * 100))
    g7 = lambda: S(math.sqrt(sel('youtube_hourly', 'alpha_beta', 'n_configs').value), f'sqrt({SEL} n_configs)', (SEL,))
    t('slices of the $7\\times7$ grid on the test part', D('grid size', g7))
    t('(last 70\\% of the evaluation windows): one-dimensional slices of the $7\\times7$ grid, each coefficient varied with the other fixed at 1.',
      P(ntest), D('grid size', g7), D('fixed value of the other coefficient'))
    t('across the 49 settings their spread is at most 0.0104 and 0.0130', D('grid settings', lambda: sel('youtube_hourly', 'alpha_beta', 'n_configs')), R(lambda: grid_spread('ndcg@10_mean')), R(lambda: grid_spread('rsi@10_mean')))
    t('with $\\alpha=\\beta=0$ (trend only), $\\Delta$Rank is 87.98, 36.21, 27.14 and 15.74 on YouTube and the hourly, 30-minute and 5-minute taxi data, against 38.08, 20.83, 12.83 and 7.53 at $(1,1)$',
      D('grid point'), *[R(lambda sc=sc: grid(sc, 0, 0, 'robustness_distortion_mean')) for sc in M4], D('slot', slot('taxi_30min')), D('slot', slot('taxi_5min')),
      *[R(lambda sc=sc: grid(sc, 1, 1, 'robustness_distortion_mean')) for sc in M4], D('grid point'), D('grid point'))
    t('$\\Delta$Rank is 38.02--43.80 at $\\alpha+\\beta=2$ and 54.34--64.12 at $\\alpha+\\beta=1$ on YouTube (87.98 without both terms), and 19.95--21.82, 27.16--28.50 and 36.21 on the hourly taxi data',
      R(lambda: grid_total('youtube_hourly', 2, min)), R(lambda: grid_total('youtube_hourly', 2, max)), D('alpha+beta'),
      R(lambda: grid_total('youtube_hourly', 1, min)), R(lambda: grid_total('youtube_hourly', 1, max)), D('alpha+beta'),
      R(lambda: grid_total('youtube_hourly', 0, min)),
      R(lambda: grid_total('taxi_hourly', 2, min)), R(lambda: grid_total('taxi_hourly', 2, max)),
      R(lambda: grid_total('taxi_hourly', 1, min)), R(lambda: grid_total('taxi_hourly', 1, max)), R(lambda: grid_total('taxi_hourly', 0, min)))
    t('The rule picked $(0.25,2)$, $(2,1.5)$, $(1.5,1.5)$ and $(1,2)$', *sum([[R(lambda sc=sc: sel(sc, 'alpha_beta', 'selected_alpha')), R(lambda sc=sc: sel(sc, 'alpha_beta', 'selected_beta'))] for sc in M4], []))
    dsel = lambda sc, c: sel(sc, 'alpha_beta', f'test_{c}_selected') - sel(sc, 'alpha_beta', f'test_{c}_default')
    t('relative to $(1,1)$, RSI@10 changed by $-0.0038$ to $+0.0033$ and NDCG@10 fell by $0.0016$ to $0.0046$',
      D('default point'), D('default point'), D('metric cut-off K=10'),
      R(lambda: smin([dsel(sc, 'rsi@10') for sc in M4])), R(lambda: smax([dsel(sc, 'rsi@10') for sc in M4])), D('metric cut-off K=10'),
      R(lambda: smin([-dsel(sc, 'ndcg@10') for sc in M4])), R(lambda: smax([-dsel(sc, 'ndcg@10') for sc in M4])))
    t('the same rule picked $J=5$ for YouTube, $J=2$ for hourly and 30-minute taxi data and $J=3$ for 5-minute data',
      R(lambda: sel('youtube_hourly', 'J', 'selected_J')), R(lambda: smax([sel(sc, 'J', 'selected_J') for sc in ['taxi_hourly', 'taxi_30min']])), D('slot', slot('taxi_30min')),
      R(lambda: sel('taxi_5min', 'J', 'selected_J')), D('slot', slot('taxi_5min')))
    t('from 1 January 1998 to 12 October 2023', D('start day', lambda: mp_date('start', 'day')), D('start year', lambda: mp_date('start', 'year')),
      D('end day', lambda: mp_date('end', 'day')), D('end year', lambda: mp_date('end', 'year')))
    t('(47 days in 1997)', Raw('longest gap without ratings in 1997: 1997-07-24 to 1997-09-08 (UTC days), counted read-only from data/raw/movielens/ratings.csv in chat 29; no result file; kept by decision of chat 29'), X('year'))
    gp = lambda fn: fn([Q(GP, c, file=f) for f in ['daily', 'weekly'] for c in ['wavelet_W64_min32_eligible_median', 'baseline_W7_min3_eligible_median']])
    t('each window ranks about 1,000 to 4,100 movies (median)', R(lambda: gp(smin), op='approx'), R(lambda: gp(smax), op='approx'))
    # Section 4.3, MovieLens sentences (T4.12)
    t('DTCWT+AF on daily data (0.8672, 0.003 above WSPI) and DWT+AF on weekly data (0.9768, 0.014 above WSPI)',
      R(lambda: V('movielens_daily', 'DTCWT+AF', NDCG)), R(lambda: V('movielens_daily', 'DTCWT+AF', NDCG) - V('movielens_daily', 'WSPI', NDCG)),
      R(lambda: V('movielens_weekly', 'DWT+AF', NDCG)), R(lambda: V('movielens_weekly', 'DWT+AF', NDCG) - V('movielens_weekly', 'WSPI', NDCG)))
    t('granularities (0.5711 and 0.8131; on weekly data 0.0003 above DTCWT+AF)', R(lambda: V('movielens_daily', 'WSPI', RHO)), R(lambda: V('movielens_weekly', 'WSPI', RHO)),
      R(lambda: V('movielens_weekly', 'WSPI', RHO) - V('movielens_weekly', 'DTCWT+AF', RHO)))
    t('on daily MovieLens data (at most 0.5711)', R(lambda: smax([V('movielens_daily', m, RHO) for m in ALL9])))
    t('and 54\\% of the ratings in this period', P(lambda: Q(YR, 'share_on_user_first_day', year='all')))
    t('has a median of 8 ratings', R(lambda: Q(GP, 'count_rank10_per_slot_median', file='daily')))
def rules_main_text_4(E):
    t = lambda anchor, *specs, **kw: E.text('main', anchor, list(specs), **kw)
    rmin, rmax = (lambda: ratio_dr(smin)), (lambda: ratio_dr(smax))
    md = lambda m, fn: fn([resp(sc, m, 'median_delay') for sc in M4])
    m3 = ['RRD', 'VSE', 'CompoundPop']
    miss3 = lambda fn: fn([resp(sc, m, 'miss_rate', 'equal64') for sc in M4 for m in m3])
    t('slot $t_0\\ge 32$', D('first common window (entry rule)', lambda: Q(PS, 'first_padded', scenario='youtube_hourly', method='WSPI')))
    t('The four scenarios contain 93, 2,632, 2,678 and 899 such entries', *[R(lambda sc=sc: resp(sc, 'WSPI', 'n_events')) for sc in M4])
    t('Its median delay is 4 to 5 slots, against 1 to 2 for AF, and it misses 16\\% to 31\\%',
      R(lambda: md('WSPI', smin)), R(lambda: md('WSPI', smax)), R(lambda: md('AF', smin)), R(lambda: md('AF', smax)),
      P(lambda: smin([resp(sc, 'WSPI', 'miss_rate') for sc in M4])), P(lambda: smax([resp(sc, 'WSPI', 'miss_rate') for sc in M4])))
    t('Its mean delay is 1.8 to 2.8 slots longer than that of AF, and 0.2 to 0.6 slots longer than that of DTCWT+AF',
      R(lambda: smin([RET_(sc, 'AF') for sc in M4])), R(lambda: smax([RET_(sc, 'AF') for sc in M4])),
      R(lambda: smin([RET_(sc, 'DTCWT+AF') for sc in M4])), R(lambda: smax([RET_(sc, 'DTCWT+AF') for sc in M4])))
    t('With $N=64$, RRD, VSE and CompoundPop miss 33\\% to 55\\%', D('window length of the equal-window configuration'), P(lambda: miss3(smin)), P(lambda: miss3(smax)))
    t('faster than all three, by 0.6 to 1.6 slots', R(lambda: smin([-RET_(sc, m, 'equal64') for sc in M4[1:] for m in m3])), R(lambda: smax([-RET_(sc, m, 'equal64') for sc in M4[1:] for m in m3])))
    t('kept it out of its Top-10 for 14 slots', D('Top-10'), R(lambda: Q(EX, 'delay_WSPI', scenario='taxi_hourly', example='worst_for_reference')))
    t('For 19.4\\% of the entries that WSPI missed on these data, the item did not meet this rule during its whole stay; the share is 32.6\\% for DTCWT+AF and 90.3\\% for DWT+AF',
      *[P(lambda m=m: resp('taxi_5min', m, 'share_miss_ineligible_whole_run')) for m in ['WSPI', 'DTCWT+AF', 'DWT+AF']])
    rf = lambda col, fn: fn([Q(RF, col, scenario=sc, config='default', method='AF') for sc in M4])
    t('is 0.24 to 0.58 for AF but only $-0.09$ to 0.18 for WSPI',
      R(lambda: rf('spearman_rsi_vs_truth_seen', smin)), R(lambda: rf('spearman_rsi_vs_truth_seen', smax)),
      R(lambda: rf('spearman_ref_rsi_vs_truth_seen', smin)), R(lambda: rf('spearman_ref_rsi_vs_truth_seen', smax)))
    t('41.88 on YouTube and 10.24, 8.72 and 4.84 on the hourly, 30-minute and 5-minute taxi data', *[R(lambda sc=sc: rb(sc, 'default', 'WSPI', 'pos_random')) for sc in M4], D('slot', slot('taxi_30min')), D('slot', slot('taxi_5min')))
    t('at least 32 observed slots', D('observed slots required', lambda: cfgv('WSPI', 'min_obs')))
    t('(default configuration; 24-hour moving average)', D('smoothing span of Figure 6'))
    t('(median of 10 repeats)', D('repeats of the benchmark', lambda: S(load(RG).repeats.min(), f'{RG} min repeats', (RG,))))
    t('($n_L=N/2^{J-1}=16$ for', X('constant in a formula'), X('constant in a formula'), D('n_L = N/2^(J-1)', S(16, '64/4')))
    t('x\\_{t-1}', X('constant in a formula'))
    t('in a median of 1.4 to 4.1~ms', R(lambda: smin([Q(RT, f'latency_ms_median_{sc}', method='WSPI') for sc in M4])), R(lambda: smax([Q(RT, f'latency_ms_median_{sc}', method='WSPI') for sc in M4])))
    rat105 = lambda: Q(RT, 'us_per_item_default_M1e4', method='WSPI') / smax([Q(RT, 'us_per_item_default_M1e4', method=m) for m in BASE6])
    t('about 105 times the time per item of the slowest baseline in our measurements', R(rat105))
    t('About 105 times the time per item of the slowest baseline', R(rat105))
    t('$+$ 2.0 to 2.8 times lower than any baseline except PFRF', R(rmin), R(rmax))
    t('$-$ Later: median delay of 4 to 5 slots against 1 to 2 for AF', R(lambda: md('WSPI', smin)), R(lambda: md('WSPI', smax)), R(lambda: md('AF', smin)), R(lambda: md('AF', smax)))
    t('the averages miss 33\\% to 55\\% of the entries', P(lambda: miss3(smin)), P(lambda: miss3(smax)))
    t('WSPI had 2.0 to 2.8 times lower $\\Delta$Rank', R(rmin), R(rmax))
    t('it scored one million items in 12.5~s', R(lambda: Q(RT, 's_all_items_default_M1e6', method='WSPI')))
    t('with a median delay of 4 to 5 slots against 1 to 2 for AF. A spike', R(lambda: md('WSPI', smin)), R(lambda: md('WSPI', smax)), R(lambda: md('AF', smin)), R(lambda: md('AF', smax)))
    t('version 2, hourly counts for 1,611 videos', D('dataset version on Kaggle'), R(lambda: yp('raw_videos')))
    t('The MovieLens 32M dataset', X('dataset name'))
    # marks for single numbers that stand for several scenarios
    for sc in ['taxi_hourly', 'taxi_30min']:
        v = sel(sc, 'J', 'selected_J').value
        E.marks.append(dict(file='main', line=0, item=f'4.9: J=2 picked for {sc}', expected='2', found=str(int(v)), status='ok' if v == 2 else 'mismatch', source=f'{SEL} selected_J'))


def rules_si_text(E):
    t = lambda anchor, *specs, **kw: E.text('si', anchor, list(specs), **kw)
    t('$N\\in\\{7,16,32,64,128\\}$. It adds', *[D('window length of the sweep', (lambda k=k: S(sorted(load(SW).window.astype(int).unique())[k], f'{SW} sorted unique window [{k}]', (SW,)))) for k in range(5)])
    t('from window 32 on', D('first window of the sweep', lambda: S(load(SW).first_window.min(), f'{SW} min first_window', (SW,))))
    t('start at $N=16$', D('smallest N of the wavelet-based methods', lambda: S(load(SW)[load(SW).method == 'WSPI'].window.astype(int).min(), f'{SW} min window of WSPI', (SW,))))
    t('with their name at $N=128$', D('largest N', lambda: S(load(SW).window.astype(int).max(), f'{SW} max window', (SW,))))
    t('(YouTube only, 0.6\\%)', P(lambda: fr('youtube_hourly', 'n_zero_energy', 'pooled') / (fr('youtube_hourly', 'n_item_windows', 'pooled') + fr('youtube_hourly', 'n_zero_energy', 'pooled'))))
    lv = lambda w, fn: S(fn(load('T3.1_level_sweep/level_sweep_summary.csv').query('window==@w').level), f'T3.1_level_sweep/level_sweep_summary.csv [window={w}] {fn.__name__} level', ('T3.1_level_sweep/level_sweep_summary.csv',))
    t('for $J=2$ to 5 with $N=64$, and $J=2$ to 4 with $N=32$', D('lowest J', lambda: lv(64, min)), D('highest J', lambda: lv(64, max)), D('window'),
      D('lowest J', lambda: lv(32, min)), D('highest J', lambda: lv(32, max)), D('window'))
    t('a collection gap of ten hours on 19 May 2018', D('gap day', lambda: ytime('gap_hours_list', 'day')), D('gap year', lambda: ytime('gap_hours_list', 'year')))
    t('with their name at $50\\times$', D('largest spike size (condition size50)'))
    t('(median of 10 repeats; band', D('repeats of the benchmark', lambda: S(load(RG).repeats.min(), f'{RG} min repeats', (RG,))))
    t('($M=10^4$)', D('number of items', lambda: S(10000, 'M of panel (b)')))
    t('(batch of $10^4$ items)', D('batch size', lambda: Q(RG, 'chunk', method='WSPI', N=64, M=10000)))
    t('shifted circularly by 0--7 slots', D('smallest shift'), D('largest shift (T3.6 design)'))
    t('(repetition 0)', D('repetition shown in panel (e)'))
    t('over 500 repetitions', D('repetitions of the synthetic test', lambda: Q(SY, 'n', shape='burst3', age='recent', kind='score', name='WSPI', metric='share_wrong')))
    t('with a 30/70 split in time', P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_tune') / sel('youtube_hourly', 'alpha_beta', 'n_common')),
      P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_test') / sel('youtube_hourly', 'alpha_beta', 'n_common')))
    t('(last 70\\% of the common windows)', P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_test') / sel('youtube_hourly', 'alpha_beta', 'n_common')))
    t('(first 30\\%)', P(lambda: sel('youtube_hourly', 'alpha_beta', 'n_tune') / sel('youtube_hourly', 'alpha_beta', 'n_common')))
    g7 = lambda: S(math.sqrt(sel('youtube_hourly', 'alpha_beta', 'n_configs').value), f'sqrt({SEL} n_configs)', (SEL,))
    t('the full $7\\times7$ grid', D('grid size', g7))
    t('in over 99\\% of windows', P(lambda: smin([V('movielens_daily', m, NDCG, col='ties_top21_share') for m in ['RRD', 'VSE']]), op='gt'))
    t('over 28 days (daily data, top) and 13 weeks', D('rolling span of Figure S7'), D('rolling span of Figure S7'))
    t('$\\alpha=2/(N+1)$). The accuracy', X('formula'), X('formula'))
    t('EWMA-eq: EWMA with $\\alpha=2/(N+1)$', X('formula'), X('formula'))
# ------------------------------------------ formulas, references, design
MATH_EVERY = [('2^{-(k-1)}', 2), ('k=1', 1), ('2^{J-1}', 2), ('x_{i,t-1}', 1), ('i=1', 1), ('\\ge 0', 1), ('t+1', 1),
              ('\\tfrac{1}{2}', 2), ('\\sqrt{-1}', 1), ('|^2', 1), ('\\frac{1}{\\log_2 (J+1)}', 2), ('\\log_2(J+1)', 1),
              ('\\ell=0', 1), ('m=0', 1), ('\\ell\\ge 1', 1), ('m\\ge 1', 1), ('\\ell=1', 1), ('[0,1]', 2), ('[-1,1]', 2),
              ('(1-R)', 1), ('(1+R-W_E)', 1), ('R(1-W_E)', 1), ('O(1)', 1), ('H^2', 1), ('N^2', 1), ('$+1$', 1),
              ('2^{\\,\\ell-1}', 2), ('2^{-(n_L-k)}', 1), ('$2N$', 1), ('+1$)', 1)]


def rules_math(E):
    for a, n in MATH_EVERY:
        for tag in ('main', 'si'):
            if E._locate(tag, a):
                E.text(tag, a, [X('constant in a formula')] * n, rule='formula', every=True)


def read_aux(path):
    """Section, table and figure numbers of the built main paper."""
    txt = Path(path).read_text(encoding='utf-8', errors='replace')
    secs = set(re.findall(r'\\numberline \{([0-9.]+)\}', txt))
    labels = dict(re.findall(r'\\newlabel\{((?:tab|fig):[^}]*)\}\{\{([0-9]+)\}', txt))
    tabs = {v for k, v in labels.items() if k.startswith('tab:')}
    figs = {v for k, v in labels.items() if k.startswith('fig:')}
    return secs, tabs, figs


def set_ref(t, ok, what, rule):
    t.rule, t.category = rule, 'reference'
    t.source = what
    t.expected = 'exists' if ok else 'missing'
    t.status = 'structural' if ok else 'mismatch'


def rules_refs(E, secs, tabs, figs, nbib):
    SEC = re.compile(r'(?:Sections?|Subsections?)(?:~|\\ | )+(\d+(?:\.\d+)?(?:(?:\s*,\s*|\s+and\s*~?|~and~|\s+to\s+)\d+(?:\.\d+)?)*)')
    TF = re.compile(r'(Tables?|Figures?)~(\d+)(?:(?:~and~|\s+and~?)(\d+))?')
    CIT = re.compile(r'\[(\d+(?:,\d+)*)\]')
    for tag, ls in E.lines.items():
        for i, l in enumerate(ls):
            math = [m.span() for m in re.finditer(r'\$[^$]*\$', l)]
            inmath = lambda p: any(a <= p < b for a, b in math)
            for m in SEC.finditer(l):
                for t in E.toks:
                    if t.file == tag and t.line == i + 1 and m.start(1) <= t.col < m.end(1) and t.status == 'unchecked':
                        set_ref(t, t.text in secs, 'section of the main paper (aux)', 'section reference')
            for m in TF.finditer(l):
                kind = tabs if m.group(1).startswith('Table') else figs
                for t in E.toks:
                    if t.file == tag and t.line == i + 1 and m.start(2) <= t.col < m.end() and t.status == 'unchecked':
                        set_ref(t, t.text in kind, f'{m.group(1).rstrip("s").lower()} number of the main paper (aux)', 'table/figure reference')
            for m in CIT.finditer(l):
                if inmath(m.start()):
                    continue
                for t in E.toks:
                    if t.file == tag and t.line == i + 1 and m.start(1) <= t.col < m.end(1) and t.status == 'unchecked':
                        set_ref(t, 1 <= t.value <= nbib and t.value == int(t.value), f'citation; {nbib} references', 'citation')


def rules_design(E):
    p = E.pattern
    w64 = lambda: cfgv('WSPI', 'window_slots')
    w7 = lambda: cfgv('AF', 'window_slots')
    lvl = lambda: cfgv('WSPI', 'level_J')
    p(r'@(10)\b', D('cut-off K=10 of NDCG@K and RSI@K (Table 4)'), rule='metric name')
    p(r'Top-(10|20)\b', D('size of the Top-K set (K=10; entry rule Top-20)'), rule='Top-K')
    p(r'[Tt]op (21)\b', D('ties counted among the top 21 scores (ties_top21_share)'), rule='tie share')
    p(r'(95)\\%', D('95% confidence level of the block bootstrap (T1.6)'), rule='confidence level')
    p(r'(7)-slot', D('baseline window length', w7), rule='window')
    p(r'(64)-slot', D('wavelet window length', w64), rule='window')
    p(r'(64) slots', D('window length', w64), rule='window')
    p(r'(7) slots', D('baseline window length', w7), rule='window')
    p(r'N=(64)\b', D('window length', w64), rule='window')
    p(r'N=(7)\b', D('baseline window length', w7), rule='window')
    p(r'J=(3)\b', D('decomposition level', lvl), rule='level')
    p(r'K=(10)\b', D('cut-off K=10'), rule='K')
    p(r'\\alpha=\\beta=(1)\b', D('default weights (Table 3)'), rule='weights')
    p(r'\\alpha=(0\.05)', D('significance level of the tests'), rule='alpha level')
    p(r'(10)\\times', D('spike size, 10 x the mean (protocol, Table 4)'), rule='spike size')
    p(r'(50) (?:low-activity|items)', D('50 target items of the robustness test (seed 42)'), rule='targets')
    p(r'(30)-minute', D('slot length (min)', slot('taxi_30min')), rule='granularity')
    p(r'(5)-minute', D('slot length (min)', slot('taxi_5min')), rule='granularity')
    p(r'(30) minutes', D('slot length (min)', slot('taxi_30min')), rule='granularity')
    p(r'(5) minutes', D('slot length (min)', slot('taxi_5min')), rule='granularity')
    p(r'Taxi (30)m\b', D('slot length (min)', slot('taxi_30min')), rule='granularity')
    p(r'Taxi (5)m\b', D('slot length (min)', slot('taxi_5min')), rule='granularity')
    p(r'(10\^\{?[46]\}?)', D('number of items or batch size of the benchmark'), rule='benchmark size')
    p(r'Stage[~ ](\d)', X('stage number of the pipeline'), rule='stage')
    p(r'Feature (\d)', X('feature number'), rule='feature')
    p(r'Algorithm (1)', X('algorithm number'), rule='algorithm')
    p(r'^(\d+): ', X('line number of Algorithm 1'), rule='algorithm line')
    p(r'Eq\.\\ \((\d)\)', X('equation number in Algorithm 1 (checked in chat 19)'), rule='equation')
    p(r'Panel~\((\w)\)', X('panel'), rule='panel')
    E.text('main', 'Energy-concentration weight & $\\alpha$ & 1.0', [D('default alpha (methods/wspi_assessment.py)')])
    E.text('main', 'Wavelet-entropy weight & $\\beta$ & 1.0', [D('default beta (methods/wspi_assessment.py)')])
    E.text('main', 'Decomposition level & $J$ & 3', [D('decomposition level', lvl)])
    p(r'T_\{\\mathrm\{future\}\}=(1)', D('evaluation horizon, one slot (protocol V5)'), rule='horizon')
    p(r'\((1)h\)', D('slot of one hour'), rule='granularity')
    p(r'\((1)[dw]\)', D('slot of one day or week'), rule='granularity')


def si_table_files(si_dir, sources):
    """The \\input tables of the SI are files made by programs with their own controls; check that
    each file is identical to the output of its program."""
    rows = []
    for f in sorted(Path(si_dir).glob('*.tex')):
        src = None
        for d in sources:
            if (Path(d) / f.name).exists():
                src = Path(d) / f.name
                break
        a = hashlib.md5(f.read_bytes()).hexdigest()
        b = hashlib.md5(src.read_bytes()).hexdigest() if src else ''
        rows.append(dict(file='si/' + f.name, md5=a, generator_output=('/'.join(src.parts[src.parts.index('Response'):]) if 'Response' in src.parts else str(src)) if src else 'not found', same=a == b))
    return rows
# ------------------------------------------------------------------ main
def main():
    global RESULTS
    ap = argparse.ArgumentParser(description='Check every number in the V5 paper and SI against the result files (T4.11)')
    ap.add_argument('--main', required=True, type=Path, help='WSPI_ScientificReports.tex of a built copy (the .aux is read next to it)')
    ap.add_argument('--si', required=True, type=Path, help='WSPI_SI.tex of the same built copy')
    ap.add_argument('--results', required=True, type=Path, help='results/revision_v5')
    ap.add_argument('--figures-json', required=True, type=Path, help='Response/Figures/paper_print/export_paper_figures_run.json (Matplotlib version)')
    ap.add_argument('--si-sources', nargs='+', required=True, type=Path, help='folders with the generator outputs of the SI tables (Response/Tables/SI, Response/Tables)')
    ap.add_argument('--out', required=True, type=Path, help='CSV report (one row per number, mark and SI table file)')
    a = ap.parse_args()
    RESULTS = a.results

    E = Engine({'main': a.main, 'si': a.si})
    secs, tabs, figs = read_aux(a.main.with_suffix('.aux'))
    nbib = len(re.findall(r'\\bibitem\{', a.main.read_text(encoding='utf-8')))
    bench = load(BJ)['hardware']['versions']
    versions = {'python': bench['python'], 'numpy': bench['numpy'], 'pandas': bench['pandas'], 'scipy': bench['scipy'],
                'pywavelets': bench['pywt'], 'dtcwt': bench['dtcwt'],
                'matplotlib': json.loads(a.figures_json.read_text(encoding='utf-8'))['matplotlib']}

    # 1. tables, cell by cell
    table_main(E, 'tab:main_default', 'default')
    table_main(E, 'tab:main_equal64', 'equal64')
    table_data(E)
    table_config(E)
    table_libs(E, versions)
    table_runtime(E)
    table_ablation(E)
    table_sens(E)
    table_resp(E)
    table_compare(E)
    table_strengths(E, secs)
    # 2. numbers in the text
    rules_main_text_1(E)
    rules_main_text_2(E)
    rules_main_text_3(E)
    rules_main_text_4(E)
    rules_si_text(E)
    # inline software versions
    for tag, ls in E.lines.items():
        for i, l in enumerate(ls):
            for m in re.finditer(r'(Python|NumPy|dtcwt) (\d+\.\d+\.\d+)', l):
                for t in E.toks:
                    if t.file == tag and t.line == i + 1 and m.start(2) <= t.col < m.end(2) and t.status == 'unchecked':
                        exp = versions[m.group(1).lower()]
                        t.rule, t.category, t.source, t.expected = 'software version', 'result', 'version recorded in the run metadata', exp
                        t.status = 'ok' if exp == m.group(2) else 'mismatch'
    # 3. formulas, references, design constants
    rules_math(E)
    rules_refs(E, secs, tabs, figs, nbib)
    rules_design(E)
    # 4. SI tables made by programs
    sit = si_table_files(a.si.parent / 'si', a.si_sources)

    rows = []
    for t in E.toks:
        rows.append(dict(kind='number', file=t.file, line=t.line, col=t.col, number=('-' if t.sign == '-' else '') + t.text,
                         category=t.category or '-', status=t.status, expected=t.expected, source=t.source, rule=t.rule,
                         context=t.ctx.strip()))
    for m in E.marks:
        rows.append(dict(kind='mark', file=m['file'], line=m['line'], col='', number=m['found'], category='mark', status=m['status'],
                         expected=m['expected'], source=m['source'], rule=m['item'], context=''))
    for s in sit:
        rows.append(dict(kind='si_table', file='si', line='', col='', number='', category='generated table', status='ok' if s['same'] else 'mismatch',
                         expected=s['md5'], source=s['generator_output'], rule=s['file'], context='cell values checked by the generator controls'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    num = [r for r in rows if r['kind'] == 'number']
    from collections import Counter
    c = Counter(r['status'] for r in num)
    print(f'numbers: {len(num)}  ' + '  '.join(f'{k}={v}' for k, v in sorted(c.items())))
    cm = Counter(r['status'] for r in rows if r['kind'] == 'mark')
    print(f'marks: {sum(cm.values())}  ' + '  '.join(f'{k}={v}' for k, v in sorted(cm.items())))
    print(f'SI table files: {len(sit)}  identical to generator output: {sum(s["same"] for s in sit)}')
    for p in E.problems:
        print('RULE PROBLEM', p)
    bad = [r for r in rows if r['status'] in ('mismatch', 'error', 'unchecked')]
    for r in bad:
        print(f"{r['status'].upper():9s} [{r['file']}:{r['line']}] {r['number']:>10s} expected {r['expected']!s:>10s} | {r['rule']} | {r['source']}")
    n_bad = len(bad) + len(E.problems)
    print('ALL NUMBERS AGREE WITH THE RESULT FILES' if n_bad == 0 else f'{n_bad} item(s) to review')
    return 0 if n_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
