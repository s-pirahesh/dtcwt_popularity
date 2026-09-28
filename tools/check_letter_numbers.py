#!/usr/bin/env python3
"""Check every number in the response letter against the result files and the built paper (task T5.1).

check_paper_numbers.py (T4.11) and check_paper_word_numbers.py (T4.11b) check the paper and its SI.
This program checks the letter, 05_Response_to_Reviewers_DRAFT.md, with the same rules and loaders
(imported, not copied). It reads the text that reaches a reader: the full version and the reviewer
version (05b is built from the same file by make_submit_letter.py, so both the [FULL-ONLY] and the
[SUBMIT-ONLY] blocks are read). Internal notes in braces {...}, the status note at the top and the task
status after "Changes in the manuscript" are skipped.

Part 1, numbers written with digits. Each number gets one category, as in the paper:
  result      recomputed from a CSV or JSON in results/revision_v5 and compared at the printed precision;
  design      a setting of the protocol (window, level, K, grid values ...), compared with the config
              or metadata file where one records it;
  reference   a section, table, figure, equation or citation of the paper, or a reviewer number;
              sections, tables and figures are checked against the .aux file of the built paper and the
              meaning the letter gives them (for example Table 15 = tab:strengths), equations against the
              equation environments of the tex, citations against the reference list.
  quote       a number inside a reviewer comment quoted in italics.
Numbers of the Supplementary Information (S1, Table S8 ...) are checked by check_si_refs.py.

Part 2, number words and quantifiers ("in all six scenarios", "in five of the six", "WSPI in three").
The claims reuse the checks of check_paper_word_numbers.py by name, plus a few letter checks.

Every number and word must be covered by exactly one rule. A rule is tied to a literal piece of one
line (its anchor). When a sentence changes, the anchor is not found and the program reports a RULE
PROBLEM instead of passing. The rules must be updated together with the sentences they hold.

The program only reads. It writes one CSV report (--out), prints a summary, and returns 1 if any number
or word differs, has no rule, or a rule does not match the text.

Example (Windows, from the project root, on a temporary built copy of V5/source):
  python tools\\check_letter_numbers.py --letter "%RESP%\\05_Response_to_Reviewers_DRAFT.md"
      --main "%TMPB%\\WSPI_ScientificReports.tex" --results results\\revision_v5
      --figures-json "%RESP%\\Figures\\paper_print\\export_paper_figures_run.json"
      --out "%RESP%\\Reports\\R33_T5.1_letter_check.csv"
"""
import sys
sys.dont_write_bytecode = True

import argparse, csv, json, math, re
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_paper_numbers as C          # noqa: E402  loaders, specs and number rules of the paper
import check_paper_word_numbers as W     # noqa: E402  word checks of the paper
import make_submit_letter as MS          # noqa: E402  the builder of the reviewer version

R, P, D, X, Raw = C.R, C.P, C.D, C.X, C.Raw
V, Q, M4, ML, M6 = C.V, C.Q, C.M4, C.ML, C.M6
BASE6, BASE5, ALL9 = C.BASE6, C.BASE5, C.ALL9
NDCG, RHO, RSI, DR = C.NDCG, C.RHO, C.RSI, C.DR
S, smin, smax = C.S, C.smin, C.smax
TAXI3 = M4[1:]

# Section titles and equation contents that the letter's references point to (used by REFS below).
LETTER_SECTIONS = {'2': 'Related Work', '3': 'Proposed Framework', '2.5': 'Research Gap', '3.1': 'Formal Problem Definition',
                   '3.2': 'Dual-Tree Complex Wavelet', '3.3': 'Why WSPI', '3.4': 'WSPI Architecture',
                   '3.5': 'Structural Feature Extraction', '3.6': 'Exponential Fusion', '4': 'Evaluation and Results',
                   '4.1': 'Evaluation Protocol and Metrics', '4.2': 'Experimental Setup and Datasets',
                   '4.3': 'Aggregate Results', '4.7': 'Computational Complexity', '4.8': 'Ablation',
                   '4.9': 'Sensitivity Analysis', '4.10': 'Failure Cases and Responsiveness'}
LETTER_EQUATIONS = {1: r'g_0(n)\approx h_0(n-\tfrac{1}{2})', 2: r'\psi_c(t)', 3: r'\mu_L=', 4: r'R=\frac{E_{\mathrm{low}}',
                    5: r'W_E=-\frac{1}{\log_2 (J+1)}', 6: r'P_{\mathrm{WSPI}}=\mu_L'}
LETTER_CITATIONS = {43: 'Harper', 44: 'YouTube'}


# ------------------------------------------------------------------ the text that reaches a reader
def letter_lines(path):
    """Lines of the letter, with internal notes, the status note, markers and task status blanked
    (blanked, not removed, so line numbers stay those of the file)."""
    raw = Path(path).read_text(encoding='utf-8').split('\n')
    out, in_status = [], False
    for l in raw:
        tag = l.strip()
        if tag.startswith('> **Status of this draft'):
            out.append('')
            continue
        if (tag.startswith('{') and tag.endswith('}')) or tag in ('[FULL-ONLY]', '[/FULL-ONLY]', '[SUBMIT-ONLY]', '[/SUBMIT-ONLY]') \
                or tag.startswith('<div') or tag == '</div>':
            out.append('')
            continue
        out.append(MS.STATUS.sub(lambda m: ' ' * len(m.group(0)), l))
    return raw, out


NUM_RE = re.compile(r'(?P<ver>\d+\.\d+\.\d+)'
                    r'|(?P<sci>\d+(?:\.\d+)?e[-−]\d+)'
                    r'|(?P<thou>(?<![,\d])\d{1,3}(?:,\d{3})+(?!,?\d)(?:\.\d+)?)'
                    r'|(?P<plain>\d+(?:\.\d+)?)')


def tokenize(lines, tag):
    toks = []
    for i, s in enumerate(lines):
        for m in NUM_RE.finditer(s):
            a, b = m.span()
            prev = s[a - 1] if a > 0 else ' '
            if prev.isalpha() or prev.isdigit() or prev in '._\\':
                continue          # R1.4, S6, db4, MD5, count_observation ...
            kind, t = m.lastgroup, m.group(0)
            if kind == 'ver':
                value, nd = 0.0, 0
            elif kind == 'sci':
                mant, exp = re.split('e', t.replace('−', '-'))
                value, nd = float(mant) * 10 ** int(exp), (len(mant.split('.')[1]) if '.' in mant else 0) - int(exp)
            else:
                value, nd = C.parse_number('plain', t)
            k = a - 1
            while k >= 0 and s[k] == ' ':
                k -= 1
            sign = ''
            if k >= 0 and s[k] in '-−+' and (k == 0 or not (s[k - 1].isalnum() or s[k - 1] in ')_}')):
                sign = '-' if s[k] in '-−' else '+'
                if sign == '-':
                    value = -value
            toks.append(C.Tok(tag, i + 1, a, b, t, value, nd, sign, s[max(0, a - 70):b + 50]))
    return toks


class LEngine(C.Engine):
    """The number engine of the paper on the lines of the letter. Reference and design rules run first;
    a text rule then gives specs to the numbers of its anchor that are still unchecked."""

    def __init__(self, lines):
        self.lines = {'L': lines}
        self.toks = tokenize(lines, 'L')
        self.problems, self.marks = [], []

    def text(self, tag, anchor, specs, rule=None, every=False):
        hits = self._locate(tag, anchor)
        rule = rule or anchor[:60]
        if len(hits) != 1:
            self.problems.append(f'[letter] anchor found {len(hits)} times: {anchor!r}')
            return
        i, j = hits[0]
        span = [t for t in self.toks if t.line == i + 1 and j <= t.col and t.end <= j + len(anchor) and t.status == 'unchecked']
        if len(span) != len(specs):
            self.problems.append(f'[letter:{i + 1}] {len(span)} unchecked numbers but {len(specs)} specs in {anchor!r} -> {[t.text for t in span]}')
            return
        for t, sp in zip(span, specs):
            C.apply_spec(t, sp, rule)


def set_status(t, ok, rule, cat, source, expected=''):
    t.rule, t.category, t.source = rule, cat, source
    t.expected = expected
    t.status = ('structural' if cat == 'reference' else 'quote' if cat == 'quote' else 'design') if ok else 'mismatch'


# ------------------------------------------------------------------ references
def read_paper(main_tex):
    aux = main_tex.with_suffix('.aux').read_text(encoding='utf-8', errors='replace')
    labels = {k: int(v) for k, v in re.findall(r'\\newlabel\{((?:tab|fig):[^}]*)\}\{\{([0-9]+)\}', aux)}
    secs = dict(re.findall(r'\\contentsline \{(?:sub)?section\}\{\\numberline \{([0-9.]+)\}([^}]*)\}', aux))
    tex = main_tex.read_text(encoding='utf-8')
    eqs = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', tex, re.S)
    bib = re.split(r'\\bibitem\{', tex.split(r'\begin{thebibliography}')[1])[1:]
    return labels, secs, eqs, bib


# Every section, table, figure, equation and citation number of the letter, as (the text before it and the
# number, its target, how many times this text occurs). A reference whose text is not listed stays
# unchecked, and a count that differs is a RULE PROBLEM, so a changed number is always reported. The
# target is compared with the built paper: tables and figures by label (.aux), sections by title (.aux),
# equations by the content of the n-th equation environment, citations by the text of the n-th bibitem.
REFS = [
    ('notation table (Section 3.1', 'sec:Formal Problem Definition', 1),
    ('table (Section 3.1, Table 2', 'tab:notation', 1),
    ('strengths-and-limitations table (Table 15', 'tab:strengths', 1),
    ('; Section 4.3', 'sec:Aggregate Results', 4),
    ('; Discussion, Table 15', 'tab:strengths', 1),
    ('Introduction and Section 2.5', 'sec:Research Gap', 2),
    ('grid than DWT (Section 4.8', 'sec:Ablation', 1),
    ('smoothing methods (Section 2.5', 'sec:Research Gap', 1),
    ('methods (Section 2.5, Table 1', 'tab:compare', 1),
    ('Section 2.5 with Table 1', 'tab:compare', 2),
    ('; Section 4.8', 'sec:Ablation', 5),
    ('configuration table (Table 6', 'tab:config', 1),
    ('manuscript:** Section 3.6', 'sec:Exponential Fusion', 3),
    ('; Section 4.9', 'sec:Sensitivity Analysis', 4),
    ('; Section 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('to the data". Section 4.9', 'sec:Sensitivity Analysis', 1),
    ('both paragraphs and Table 13', 'tab:sens', 1),
    ('robustness study (Section 4.3', 'sec:Aggregate Results', 1),
    ('manuscript:** Section 4.1', 'sec:Evaluation Protocol and Metrics', 2),
    ('perturbations" in Section 4.3', 'sec:Aggregate Results', 1),
    ('more accurate (Sections 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('accurate (Sections 4.2 and 4.3', 'sec:Aggregate Results', 1),
    ('Sections 4.2 and 4.3, Tables 8', 'tab:main_default', 1),
    ('and 4.3, Tables 8 and 9', 'tab:main_equal64', 1),
    ('Tables 8 and 9, Figure 6', 'fig:movielens', 1),
    ('manuscript:** Section 4.2', 'sec:Experimental Setup and Datasets', 4),
    ('Section 4.2 and Table 5', 'tab:data', 1),
    ('main results: Section 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('paragraph) and Section 4.3', 'sec:Aggregate Results', 1),
    ('and Section 4.3 (Tables 8', 'tab:main_default', 1),
    ('Section 4.3 (Tables 8 and 9', 'tab:main_equal64', 1),
    ('9, results text, Figure 6', 'fig:movielens', 1),
    ('availability and reference [43', 'cite:Harper', 1),
    ('**Response.** Section 4.7', 'sec:Computational Complexity', 1),
    ('Core Ultra 7 255H; Table 11', 'tab:runtime', 1),
    ('manuscript:** Section 4.7', 'sec:Computational Complexity', 2),
    ('complexity paragraph, Table 10', 'tab:complexity', 1),
    ('Table 10 and new Table 11', 'tab:runtime', 1),
    ('taxi scenarios (Section 3.5', 'sec:Structural Feature Extraction', 1),
    ('on its split (Section 4.9', 'sec:Sensitivity Analysis', 1),
    ('manuscript:** Section 3.5', 'sec:Structural Feature Extraction', 2),
    ('contribution, and Section 3.3', 'sec:Why WSPI', 1),
    ('Discussion and in Table 15', 'tab:strengths', 1),
    ('number of items in Section 3.1', 'sec:Formal Problem Definition', 1),
    ('table was added (Table 2', 'tab:notation', 1),
    ('manuscript:** Section 3.1', 'sec:Formal Problem Definition', 4),
    ('Section 3.1 and Table 2', 'tab:notation', 1),
    ('complex-wavelet equation (now Eq. (2', 'eq:2', 1),
    ('feature equations (now Eqs. (3', 'eq:3', 1),
    ('equations (now Eqs. (3)–(5', 'eq:5', 1),
    ('text after them; Table 3', 'tab:hyper', 1),
    ('Algorithm 1 (input); Section 4.7', 'sec:Computational Complexity', 1),
    ('. Figure 1', 'fig:pipeline', 1),
    ('the symbols of Section 3', 'sec:Proposed Framework', 1),
    ('preprocessing of Section 3.4', 'sec:WSPI Architecture', 1),
    ('**Response.** Section 3.2', 'sec:Dual-Tree Complex Wavelet', 1),
    ('low-pass filters (new Eq. (1', 'eq:1', 1),
    ('property directly (Section 4.8', 'sec:Ablation', 1),
    ('manuscript:** Section 3.2', 'sec:Dual-Tree Complex Wavelet', 1),
    ('condition as a new Eq. (1', 'eq:1', 1),
    ('**Response.** Agreed. Section 3.1', 'sec:Formal Problem Definition', 1),
    ('for each, and Section 4.1', 'sec:Evaluation Protocol and Metrics', 1),
    ('; Section 4.1', 'sec:Evaluation Protocol and Metrics', 1),
    ('manuscript:** Section 4.3', 'sec:Aggregate Results', 3),
    ('manuscript:** Section 4.8', 'sec:Ablation', 3),
    ('Section 4.8 and Table 12', 'tab:ablation', 1),
    ('run is listed in Table 6', 'tab:config', 1),
    ('**Response.** Added as Table 15', 'tab:strengths', 1),
    ('manuscript:** new Table 15', 'tab:strengths', 1),
    ('**Response.** Section 3.1', 'sec:Formal Problem Definition', 1),
    ('paragraphs) and new Table 2', 'tab:notation', 1),
    ('Introduction to Section 4', 'sec:Evaluation and Results', 1),
    ('now refers to Section 3.1', 'sec:Formal Problem Definition', 1),
    ('hourly taxi data (Section 4.8', 'sec:Ablation', 1),
    ('Section 4.8). Section 2', 'sec:Related Work', 1),
    ('contributions) and Section 2.5', 'sec:Research Gap', 1),
    ('feature measures (Section 3.5', 'sec:Structural Feature Extraction', 1),
    ('of 1.7 to 2.5 (Section 4.8', 'sec:Ablation', 1),
    ('; Section 3.5', 'sec:Structural Feature Extraction', 1),
    ('R1.2 and R1.3: Section 3.6', 'sec:Exponential Fusion', 1),
    ('manuscript:** see R1.6: Section 4.1', 'sec:Evaluation Protocol and Metrics', 1),
    ('; Sections 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('; Sections 4.2 and 4.3', 'sec:Aggregate Results', 1),
    ('in two parts (Section 4.8', 'sec:Ablation', 1),
    ('; Section 3.2', 'sec:Dual-Tree Complex Wavelet', 1),
    ('first point of Section 3.3', 'sec:Why WSPI', 1),
    ('a single value. Table 10', 'tab:complexity', 1),
    ('manuscript:** Section 4.7, Table 10', 'tab:complexity', 1),
    ('4.7, Table 10 and Table 11', 'tab:runtime', 1),
    ('two paragraphs and Table 12', 'tab:ablation', 1),
    ('**Response.** Thank you. Section 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('dataset cited as reference [44', 'cite:YouTube', 1),
    ('reference [44]), Table 5', 'tab:data', 1),
    ('coefficients, and Section 3.4', 'sec:WSPI Architecture', 1),
    ('manuscript:** Section 3.4', 'sec:WSPI Architecture', 1),
    ('subsection (Subsection 4.10', 'sec:Failure Cases and Responsiveness', 1),
    ('Subsection 4.10, Table 14', 'tab:resp', 1),
    ('4.10, Table 14 and Figure 10', 'fig:surge', 1),
    ('manuscript:** new Subsection 4.10', 'sec:Failure Cases and Responsiveness', 1),
    ('responsiveness) with Table 14', 'tab:resp', 1),
    ('with Table 14 and Figure 10', 'fig:surge', 1),
    ('perturbations" of Section 4.3', 'sec:Aggregate Results', 1),
    ('the Discussion, Table 15', 'tab:strengths', 1),
    ('. Section 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('μ_L + R − W_E (Section 4.8', 'sec:Ablation', 1),
    ('method is listed in Table 6', 'tab:config', 1),
    ('; Section 4.7', 'sec:Computational Complexity', 1),
    ('; tables of Section 4.2', 'sec:Experimental Setup and Datasets', 1),
    ('and tables of Section 4', 'sec:Evaluation and Results', 1),
]

REF_RX = [('section', re.compile(r'(?:Sections?|Subsections?) ((?:\d+(?:\.\d+)?)(?:(?:, | and | to |–)\d+(?:\.\d+)?)*)')),
          ('table', re.compile(r'(?<!Supplementary )Tables? ((?:\d+)(?:(?:, | and |–)\d+)*)')),
          ('figure', re.compile(r'(?<!Supplementary )(?:Figures?|Figs?\.) ((?:\d+)(?:(?:, | and |–)\d+)*)')),
          ('equation', re.compile(r'Eqs?\. \((\d+)\)(?:–\((\d+)\))?')),
          ('citation', re.compile(r'\[(\d+)\]'))]


def ref_ok(target, n, labels, secs, eqs, bib):
    kind, _, want = target.partition(':')
    if kind in ('tab', 'fig'):
        got = labels.get(target)
        return got == int(float(n)), f'{target} = {"table" if kind == "tab" else "figure"} {got} (aux)'
    if kind == 'sec':
        title = secs.get(n, '')
        return bool(title) and want in title, f'section {n} of the paper (aux): {title[:50] or "missing"}'
    if kind == 'eq':
        k = int(n)
        snip = LETTER_EQUATIONS.get(k)
        return snip is not None and k <= len(eqs) and snip in eqs[k - 1], f'equation environment {k} of the tex contains {snip!r}'
    k = int(n)
    return k <= len(bib) and want in bib[k - 1], f'reference [{k}] of the paper ({len(bib)} references) mentions {want!r}'


def rules_refs(E, labels, secs, eqs, bib):
    lines = E.lines['L']
    cand = {}
    for i, l in enumerate(lines):
        for kind, r in REF_RX:
            for m in r.finditer(l):
                a, b = (m.start(1), m.end(2) if m.group(2) else m.end(1)) if kind == 'equation' else m.span(1)
                for t in E.toks:
                    if t.line == i + 1 and a <= t.col < b and t.status == 'unchecked':
                        cand[id(t)] = t
    done = set()
    for left, target, n in sorted(REFS, key=lambda x: -len(x[0])):
        hits = []
        for i, l in enumerate(lines):
            for m in re.finditer(re.escape(left), l):
                t = next((t for t in cand.values() if t.line == i + 1 and t.end == m.end() and id(t) not in done), None)
                if t is not None:
                    hits.append(t)
        if len(hits) != n:
            E.problems.append(f'[letter] reference text found {len(hits)} times (want {n}): {left!r}')
        for t in hits:
            done.add(id(t))
            ok, src = ref_ok(target, t.text, labels, secs, eqs, bib)
            set_status(t, ok, 'reference: ' + left[-40:], 'reference', src, target)
    for i, l in enumerate(lines):
        m = re.match(r'^## Reviewer (\d)', l)
        if m:
            t = next(t for t in E.toks if t.line == i + 1 and t.col == m.start(1))
            set_status(t, 1 <= t.value <= 4, 'reviewer number', 'reference', 'four reviewers')


def rules_quotes(E):
    """Numbers inside the quoted reviewer comments (the italic text after an item number)."""
    rx = re.compile(r'^\*\*(?:ED\.\d+|R\d\.\d+)\*\* \*(.*?)\*$')
    for i, l in enumerate(E.lines['L']):
        m = rx.match(l)
        if not m:
            continue
        for t in E.toks:
            if t.line == i + 1 and m.start(1) <= t.col < m.end(1) and t.status == 'unchecked':
                set_status(t, True, 'reviewer comment', 'quote', 'quoted from the reviewer comment')


def rules_notation(E):
    """Metric names (NDCG@10, RSI@10) and Top-10 are notation, never results; they are classified before the
    sentence rules so that the anchors need not list them."""
    E.pattern(r'@(10)\b', D('cut-off K=10 of NDCG@K and RSI@K (Table 4)'), rule='metric name')
    E.pattern(r'Top-(10)\b', D('size of the Top-K set (K=10)'), rule='Top-K')


def rules_design(E):
    p = E.pattern
    w64 = lambda: C.cfgv('WSPI', 'window_slots')
    w7 = lambda: C.cfgv('AF', 'window_slots')
    lvl = lambda: C.cfgv('WSPI', 'level_J')
    p(r'@(10)\b', D('cut-off K=10 of NDCG@K and RSI@K (Table 4)'), rule='metric name')
    p(r'Top-(10)\b', D('size of the Top-K set (K=10)'), rule='Top-K')
    p(r'top (21) items', D('ties counted among the top 21 scores (ties_top21_share)'), rule='tie share')
    p(r'\b(95)% confidence', D('95% confidence level of the block bootstrap (T1.6)'), rule='confidence level')
    p(r'\b(7)-slot', D('baseline window length', w7), rule='window')
    p(r'\b(64)-slot', D('wavelet window length', w64), rule='window')
    p(r'\b(64) slots', D('window length', w64), rule='window')
    p(r'N = (64)\b', D('window length', w64), rule='window')
    p(r'N = (7) for', D('baseline window length', w7), rule='window')
    p(r'J = (3)\b', D('decomposition level', lvl), rule='level')
    p(r'α = β = (1)\b', D('default weights (Table 3)'), rule='weights')
    p(r'\((1), 1\)', D('default weights (alpha, beta) = (1, 1)'), rule='weights')
    p(r'\(1, (1)\)', D('default weights (alpha, beta) = (1, 1)'), rule='weights')
    p(r'\b(30)-minute', D('slot length (min)', C.slot('taxi_30min')), rule='granularity')
    p(r'\b(5)-minute', D('slot length (min)', C.slot('taxi_5min')), rule='granularity')
    p(r'T_future \(?=? ?(1)\b', D('evaluation horizon, one slot (protocol V5)'), rule='horizon')
    p(r'T_future = (1) slot', D('evaluation horizon, one slot (protocol V5)'), rule='horizon')
    for g in range(5):
        p(r'N ∈ \{(7), (16), (32), (64), (128)\}',
          D('window length of the sweep', (lambda k=g: S(sorted(C.load(C.SW).window.astype(int).unique())[k], f'{C.SW} sorted unique window [{k}]', (C.SW,)))),
          rule='sweep', group=g + 1)
    p(r'O\((1)\)', D('order of the update cost'), rule='big O')
    p(r'Stage (\d)', X('stage number of the pipeline (Section 3.4)'), rule='stage')


# ------------------------------------------------------------------ word engine on the letter
class LWEngine(W.Engine):
    def __init__(self, lines):
        self.lines, self.toks, self.problems, self.extra, self.body = {'L': lines}, [], [], [], {'L': (0, len(lines))}
        for i, s in enumerate(lines):
            for m in W.WORD_RE.finditer(s):
                x, y = m.span(1)
                self.toks.append(W.WTok('L', i + 1, x, y, s[x:y], s[max(0, x - 70):y + 50]))


# RULES_NUMBERS_BEGIN
IU = 'T1.5_leakage_audit/whole_file/item_universe.csv'
PT_W = 'T1.6_stats/all_paired_tests.csv'
PT_C = 'T1.6_stats/causal/all_paired_tests.csv'
LS = 'T3.1_level_sweep/level_sweep_summary.csv'
EQ64_YT = 'T2.2_window_sweep/youtube_hourly/W064/metadata/protocol_v5_run.json'
ROB_META = 'T3.5_robustness/youtube_hourly/metadata/robustness_run.json'


def lv(sc, m, J, col, N=64):
    return Q(LS, col, scenario=sc, method=m, window=N, level=J)


def lv_levels(N, fn):
    d = C.load(LS)
    return S(fn(d[d.window == N].level), f'{LS} [window={N}] {fn.__name__} level', (LS,))


def spike_param(name, key):
    sp = {x['name']: x for x in C.load(ROB_META)['spikes']}
    return S(sp[name][key], f'{ROB_META} spikes[{name}].{key}', (ROB_META,))


def noise_snr(name):
    nz = {x['name']: x for x in C.load(ROB_META)['noises']}
    return S(nz[name]['snr_db'], f'{ROB_META} noises[{name}].snr_db', (ROB_META,))


def verdicts_changed():
    k = ['run_group', 'scenario', 'metric', 'reference', 'method']
    a, b = C.load(PT_W), C.load(PT_C)
    m = a.merge(b, on=k, suffixes=('_w', '_c'))
    if len(m) != len(a) or len(m) != len(b):
        raise LookupError('the two test files do not pair one to one')
    return m, int((m.verdict_w != m.verdict_c).sum())


def ds_year(dataset, col):
    d = C.load(C.DS)
    v = d[d.dataset == dataset][col]
    if len(v) != 1:
        raise LookupError(f'{C.DS}: {len(v)} rows for {dataset}')
    return S(int(str(v.iloc[0])[:4]), f'{C.DS} [dataset={dataset}] {col} (year)', (C.DS,))


def ab_ratio(fn):
    return fn([C.ab(sc, 'Trend', DR) / C.ab(sc, 'WSPI', DR) for sc in M4])


def rules_numbers(E):
    t = lambda anchor, *specs: E.text('L', anchor, list(specs))
    w64 = lambda: C.cfgv('WSPI', 'window_slots')
    w7 = lambda: C.cfgv('AF', 'window_slots')
    lvl = lambda: C.cfgv('WSPI', 'level_J')
    ntune = lambda: C.sel('youtube_hourly', 'alpha_beta', 'n_tune') / C.sel('youtube_hourly', 'alpha_beta', 'n_common')
    ntest = lambda: C.sel('youtube_hourly', 'alpha_beta', 'n_test') / C.sel('youtube_hourly', 'alpha_beta', 'n_common')
    md = lambda m, fn: fn([C.resp(sc, m, 'median_delay') for sc in M4])
    DP = lambda: D('default point (alpha, beta) = (1, 1)')
    FORM = lambda: X('constant in a formula')
    # header and general response
    t('e86fd312-1ceb-4ca5-b5af-b57fae4a2a4a', X('submission ID'), X('submission ID'))
    for k in range(1, 7):
        E.pattern(rf'^({k})\. \*\*', X('number of a point of the general response'), rule='list number')
    t('(7 and 64 slots, with this difference stated)', D('baseline window length', w7), D('window length', w64))
    t('a 2-D α–β grid with a leakage-free 30/70 selection check', D('two coefficients alpha and beta'), P(ntune), P(ntest))
    # ED.1
    t('(63 instead of 64 slots, and 6 instead of 7)', D('window of V4 = N - 1', lambda: w64() - 1), D('window length', w64),
      D('window of V4 = N - 1', lambda: w7() - 1), D('window length', w7))
    t('DTCWT+AF was run with J = 2, while WSPI used J = 3', D('level of DTCWT+AF in the V4 runs (int(log2 30) - 2; tracker section J)'), D('level', lvl))
    t('(General response, point 2)', X('point of the general response'))
    # R1.2
    t('DTCWT leaves 16 approximation coefficients', D('n_L = N/2^(J-1)', S(16, '64 / 2^(3-1)')))
    t('(J = 2 to 5 at N = 64, and J = 2 to 4 at N = 32,',
      D('lowest J', lambda: lv_levels(64, min)), D('highest J', lambda: lv_levels(64, max)), D('window', w64),
      D('lowest J', lambda: lv_levels(32, min)), D('highest J', lambda: lv_levels(32, max)), D('window of the level sweep', S(32, 'N=32')))
    t('from J = 2 to J = 5, RSI@10 of WSPI rises from 0.9041 to 0.9802 on YouTube and from 0.8251 to 0.9505 on the hourly taxi data, while NDCG@10 falls from 0.9772 to 0.9674 and from 0.9079 to 0.8726',
      D('lowest J', lambda: lv_levels(64, min)), D('highest J', lambda: lv_levels(64, max)),
      R(lambda: lv('youtube_hourly', 'WSPI', 2, 'rsi@10_mean')), R(lambda: lv('youtube_hourly', 'WSPI', 5, 'rsi@10_mean')),
      R(lambda: lv('taxi_hourly', 'WSPI', 2, 'rsi@10_mean')), R(lambda: lv('taxi_hourly', 'WSPI', 5, 'rsi@10_mean')),
      R(lambda: lv('youtube_hourly', 'WSPI', 2, 'ndcg@10_mean')), R(lambda: lv('youtube_hourly', 'WSPI', 5, 'ndcg@10_mean')),
      R(lambda: lv('taxi_hourly', 'WSPI', 2, 'ndcg@10_mean')), R(lambda: lv('taxi_hourly', 'WSPI', 5, 'ndcg@10_mean')))
    t('choosing J on the first 30% of the evaluation windows gave J = 5 for YouTube, J = 2 for the hourly and 30-minute taxi data and J = 3 for the 5-minute data',
      P(ntune), R(lambda: C.sel('youtube_hourly', 'J', 'selected_J')),
      R(lambda: C.sel('taxi_hourly', 'J', 'selected_J')), D('slot length (min)', C.slot('taxi_30min')),
      R(lambda: C.sel('taxi_5min', 'J', 'selected_J')), D('slot length (min)', C.slot('taxi_5min')))
    t('(the wavelet-based methods from N = 16)', D('smallest N of the wavelet-based methods',
      lambda: S(C.load(C.SW)[C.load(C.SW).method == 'WSPI'].window.astype(int).min(), f'{C.SW} min window of WSPI', (C.SW,))))
    # R1.3
    t('the full grid α, β ∈ {0, 0.25, …, 2}', *[D('grid value (T3.2)', (lambda k=k: S(sorted(C.load(C.GR).alpha.unique())[k], f'{C.GR} sorted unique alpha [{k}]', (C.GR,)))) for k in (0, 1, 6)])
    t('the first 30% are used for tuning and the last 70% for testing', P(ntune), P(ntest))
    t('within 1% of the best one', D('admissible if NDCG@10 >= 0.99 x best', lambda: (1 - C.sel('youtube_hourly', 'alpha_beta', 'ndcg_threshold') / C.sel('youtube_hourly', 'alpha_beta', 'ndcg_star')) * 100))
    t('across the 49 settings their spread is at most 0.0104 and 0.0130', D('grid settings', lambda: C.sel('youtube_hourly', 'alpha_beta', 'n_configs')),
      R(lambda: C.grid_spread('ndcg@10_mean')), R(lambda: C.grid_spread('rsi@10_mean')))
    t('with α = β = 0 (trend only), ΔRank is 87.98, 36.21, 27.14 and 15.74 on YouTube and the hourly, 30-minute and 5-minute taxi data, against 38.08, 20.83, 12.83 and 7.53 at (1, 1)',
      D('grid point'), *[R(lambda sc=sc: C.grid(sc, 0, 0, 'robustness_distortion_mean')) for sc in M4], D('slot', C.slot('taxi_30min')), D('slot', C.slot('taxi_5min')),
      *[R(lambda sc=sc: C.grid(sc, 1, 1, 'robustness_distortion_mean')) for sc in M4], DP(), DP())
    t('The rule picked (0.25, 2), (2, 1.5), (1.5, 1.5) and (1, 2)', *sum([[R(lambda sc=sc: C.sel(sc, 'alpha_beta', 'selected_alpha')), R(lambda sc=sc: C.sel(sc, 'alpha_beta', 'selected_beta'))] for sc in M4], []))
    dsel = lambda sc, c: C.sel(sc, 'alpha_beta', f'test_{c}_selected') - C.sel(sc, 'alpha_beta', f'test_{c}_default')
    t('Relative to (1, 1), RSI@10 changed by −0.0038 to +0.0033 and NDCG@10 fell by 0.0016 to 0.0046',
      DP(), DP(), R(lambda: smin([dsel(sc, 'rsi@10') for sc in M4])), R(lambda: smax([dsel(sc, 'rsi@10') for sc in M4])),
      R(lambda: smin([-dsel(sc, 'ndcg@10') for sc in M4])), R(lambda: smax([-dsel(sc, 'ndcg@10') for sc in M4])))
    # R1.4
    t('the spike size (2, 5, 10, 20 and 50 times the mean)', *[D('spike size of the extended study', (lambda k=k: spike_param(f'size{k}', 'size'))) for k in (2, 5, 10, 20, 50)])
    t('(1, 3 and 6 consecutive slots)', D('duration', lambda: spike_param('size10', 'duration')), D('duration', lambda: spike_param('dur3', 'duration')), D('duration', lambda: spike_param('dur6', 'duration')))
    t('Gaussian noise at 10 dB and 0 dB SNR', D('SNR', lambda: noise_snr('noise_gauss_snr10')), D('SNR', lambda: noise_snr('noise_gauss_snr0')))
    t('the same 50 low-activity items', D('target items', lambda: S(C.load(ROB_META)['n_targets'], f'{ROB_META} n_targets', (ROB_META,))))
    # R1.5
    t('MovieLens 32M, the number', X('dataset name'))
    t('(1998–2023)', D('first year', lambda: ds_year('movielens_daily', 'first_time')), D('last year', lambda: ds_year('movielens_daily', 'last_time')))
    t('(on weekly data by only 0.0003)', R(lambda: V('movielens_weekly', 'WSPI', RHO) - V('movielens_weekly', 'DTCWT+AF', RHO)))
    # R1.6
    t('(10,000 resamples)', D('bootstrap resamples (tools/stats_report.py, T1.6)'))
    t('one day on YouTube (24 slots)', D('bootstrap block', lambda: V('youtube_hourly', 'WSPI', NDCG, col='block')))
    t('(168, 336 and 2,016 slots)', *[D('bootstrap block', (lambda sc=sc: V(sc, 'WSPI', NDCG, col='block'))) for sc in TAXI3])
    t('the block is 7 days (daily data) and 4 weeks (weekly data)', D('bootstrap block', lambda: V('movielens_daily', 'WSPI', NDCG, col='block')), D('bootstrap block', lambda: V('movielens_weekly', 'WSPI', NDCG, col='block')))
    # R1.7
    t('(Intel Core Ultra 7 255H;', X('processor model'), X('processor model'))
    t('WSPI needs 11.7 µs, about the same as DTCWT+AF (11.4 µs), and about 105 times the slowest conventional baseline',
      R(lambda: Q(C.RT, 'us_per_item_default_M1e4', method='WSPI')), R(lambda: Q(C.RT, 'us_per_item_default_M1e4', method='DTCWT+AF')),
      R(lambda: Q(C.RT, 'us_per_item_default_M1e4', method='WSPI') / smax([Q(C.RT, 'us_per_item_default_M1e4', method=m) for m in BASE6])))
    t('one million items take 12.5 s', R(lambda: Q(C.RT, 's_all_items_default_M1e6', method='WSPI')))
    t('a median of 4.1 ms (YouTube) and 1.4 ms (5-minute taxi)', R(lambda: Q(C.RT, 'latency_ms_median_youtube_hourly', method='WSPI')),
      R(lambda: Q(C.RT, 'latency_ms_median_taxi_5min', method='WSPI')), D('slot', C.slot('taxi_5min')))
    # R1.8
    t('W_E·log₂(J+1) = h(R) + (1−R)·H(q)', FORM(), FORM())
    t('the Spearman correlation of R and W_E is −0.993 to −0.998, and R explains 99.4–99.7% of the variance of W_E',
      R(lambda: smax([C.fr(sc, 'sp_R_WE') for sc in M4])), R(lambda: smin([C.fr(sc, 'sp_R_WE') for sc in M4])),
      P(lambda: smin([C.fr(sc, 'eta2_WE_given_R', 'pooled') for sc in M4])), P(lambda: smax([C.fr(sc, 'eta2_WE_given_R', 'pooled') for sc in M4])))
    # R2.1
    t('(median delay of 4 to 5 slots against 1 to 2 for AF)', R(lambda: md('WSPI', smin)), R(lambda: md('WSPI', smax)), R(lambda: md('AF', smin)), R(lambda: md('AF', smax)))
    # R3.2, R3.3
    t('(n_L = N/2^(J−1) = 16', FORM(), FORM(), D('n_L = N/2^(J-1)', S(16, '64 / 2^(3-1)')))
    t('Algorithm 1 (input)', X('algorithm number'))
    t('the 2:1 redundancy', X('redundancy of DTCWT'), X('redundancy of DTCWT'))
    # R3.4 and R3.5
    for anchor in ('its NDCG@10 is 0.1389 on YouTube and 0.4576 on the hourly taxi data', '(NDCG@10 = 0.1389 on YouTube and 0.4576 on the hourly taxi data)'):
        t(anchor, R(lambda: V('youtube_hourly', 'PFRF', NDCG)), R(lambda: V('taxi_hourly', 'PFRF', NDCG)))
    t('has RSI@10 of 0.9897 on YouTube and 0.9754 on the hourly taxi data, well above WSPI (0.9446 and 0.8888), while its NDCG@10 is lower (0.9651 and 0.8596, against 0.9715 and 0.8824)',
      *[R(lambda sc=sc, m=m, met=met: V(sc, m, met, 'equal64')) for met, m in ((RSI, 'SMA'), (RSI, 'WSPI'), (NDCG, 'SMA'), (NDCG, 'WSPI')) for sc in ('youtube_hourly', 'taxi_hourly')])
    # R3.6
    t('RSI@10 falls from 0.9446 to 0.8906 on YouTube and from 0.8888 to 0.7871 on the hourly taxi data',
      R(lambda: C.ab('youtube_hourly', 'WSPI', RSI)), R(lambda: C.ab('youtube_hourly', 'DWT-WSPI', RSI)), R(lambda: C.ab('taxi_hourly', 'WSPI', RSI)), R(lambda: C.ab('taxi_hourly', 'DWT-WSPI', RSI)))
    t('All 32 comparisons of RSI@10', R(lambda: S(len(C.load(C.ABT)[C.load(C.ABT).method.str.startswith('DWT') & C.load(C.ABT).metric.isin([RSI, DR])]), f'{C.ABT} rows WSPI vs DWT-* for RSI@10 and dRank', (C.ABT,))))
    t('(0.9814 against 0.9715 on YouTube)', R(lambda: C.ab('youtube_hourly', 'DWT-WSPI', NDCG)), R(lambda: C.ab('youtube_hourly', 'WSPI', NDCG)))
    # R3.7
    t('entry rule of 32 observed slots for the wavelet-based methods and 3 for the baselines',
      D('observed slots required', lambda: C.cfgv('WSPI', 'min_obs')), D('observed slots required', lambda: C.cfgv('AF', 'min_obs')))
    t('(50 views on YouTube, 24 trips or ratings', D('item threshold', lambda: Q(C.DS, 'min_obs', dataset='youtube_hourly')), D('item threshold', lambda: Q(C.DS, 'min_obs', dataset='taxi_hourly')))
    # R4.1
    t('x_i = (x_{i,t−N}, …, x_{i,t−1})', FORM())
    t('between the rankings at t and t+1', FORM())
    # R4.2
    t('(RSI@10 0.9422 against 0.9414 on YouTube)', R(lambda: C.ab('youtube_hourly', 'Trend', RSI)), R(lambda: lv('youtube_hourly', 'DTCWT+AF', 3, 'rsi@10_mean')))
    for anchor in ('ΔRank rises from 37.41 to 92.35 on YouTube and from 20.40 to 34.70 on the hourly taxi data (Section 4.8)',
                   'ΔRank rises from 37.41 to 92.35 on YouTube and from 20.40 to 34.70 on the hourly taxi data; each'):
        t(anchor, R(lambda: C.ab('youtube_hourly', 'WSPI', DR)), R(lambda: C.ab('youtube_hourly', 'Trend', DR)), R(lambda: C.ab('taxi_hourly', 'WSPI', DR)), R(lambda: C.ab('taxi_hourly', 'Trend', DR)))
    # R4.3
    t('(partial Spearman correlation 0.06 to 0.14)', R(lambda: smin([C.fr(sc, 'pc_S_y_mu') for sc in M4])), R(lambda: smax([C.fr(sc, 'pc_S_y_mu') for sc in M4])))
    t('by a factor of 1.7 to 2.5', R(lambda: ab_ratio(smin)), R(lambda: ab_ratio(smax)))
    t('Feature 1 (an untested', X('feature number'))
    # R4.4
    t('chooses the values on the first 30% of the evaluation windows and tests them on the last 70%', P(ntune), P(ntest))
    # R4.7
    t('circularly by 0 to 7 slots', D('smallest shift'), D('largest shift (T3.6 design)'))
    cv = lambda tr, fn: fn([Q(C.SH, 'mean', scenario=sc, transform=tr, metric='cv_E_3') for sc in M4])
    t('the energy of detail level 3 varies by 5–9% across the shifts', D('detail level index'), P(lambda: cv('DTCWT', smin)), P(lambda: cv('DTCWT', smax)))
    t('with DWT (db4) it varies by 32–38%', P(lambda: cv('DWT', smin)), P(lambda: cv('DWT', smax)))
    rr = lambda fn: fn([Q(C.SH, 'ratio_dwt_over_dtcwt', scenario=sc, transform='DWT', metric=m) for sc in M4 for m in ['sd_R', 'sd_WE']])
    t('The variation of R and W_E is 7 to 12 times smaller with DTCWT', R(lambda: rr(smin)), R(lambda: rr(smax)))
    sw = lambda name, fn: fn([Q(C.SY, 'mean', shape=sh, age='recent', kind='score', name=name, metric='share_wrong') for sh in ['spike', 'burst3']])
    t('the WSPI score falls in 7–25% of the steps', P(lambda: sw('WSPI', smin)), P(lambda: sw('WSPI', smax)))
    t('with DWT features it falls in 56–66%', P(lambda: sw('DWT-WSPI', smin)), P(lambda: sw('DWT-WSPI', smax)))
    t('DWT+AF falls in 65–75% of the steps', P(lambda: sw('DWT+AF', smin)), P(lambda: sw('DWT+AF', smax)))
    # R4.8
    t('keeps 16 real low-pass and 56 complex detail coefficients per item, 1 KiB in double precision (about 2N values)',
      D('n_L = N/2^(J-1)', S(16, '64/4')), D('detail coefficients 32+16+8', S(56, '32+16+8')),
      R(lambda: Q(C.RT, 'coef_bytes_per_item_N64', method='WSPI') / 1024), D('2N values: 128 = 16 + 2 x 56', S(2, '(16 + 2*56)/64')))
    t('about 3.2 kB per item in a batch, against 24–73 bytes for the baselines',
      R(lambda: Q(C.RT, 'traced_bytes_per_item_default_B1e4', method='WSPI') / 1000),
      R(lambda: smin([Q(C.RT, 'traced_bytes_per_item_default_B1e4', method=m) for m in BASE6])),
      R(lambda: smax([Q(C.RT, 'traced_bytes_per_item_default_B1e4', method=m) for m in BASE6])))
    # R4.9
    t('(by at most 0.0076). They lower NDCG@10 by at most 0.0042',
      R(lambda: smax([C.ab(sc, 'WSPI', RSI) - C.ab(sc, 'Trend', RSI) for sc in M4])), R(lambda: smax([C.ab(sc, 'Trend', NDCG) - C.ab(sc, 'WSPI', NDCG) for sc in M4])))
    # R4.10
    t('(CC0, version 2, file', D('dataset version on Kaggle'))
    t('about 1,500 videos uploaded in April 2018; the list was retrieved on 7 May 2018; hourly snapshots from 7 May 2018 18:00 to 5 June 2018 16:00 (1,611 videos × 695 hours)',
      X('from the Kaggle dataset description (no CSV)'), X('from the Kaggle dataset description (no CSV)'),
      D('first snapshot, day', lambda: C.ytime('raw_first_time', 'day')), D('year', lambda: C.ytime('raw_first_time', 'year')),
      D('first snapshot, day', lambda: C.ytime('raw_first_time', 'day')), D('year', lambda: C.ytime('raw_first_time', 'year')),
      D('hour', lambda: C.ytime('raw_first_time', 'hour')), D('minute', lambda: C.ytime('raw_first_time', 'minute')),
      D('last snapshot, day', lambda: C.ytime('raw_last_time', 'day')), D('year', lambda: C.ytime('raw_last_time', 'year')),
      D('hour', lambda: C.ytime('raw_last_time', 'hour')), D('minute', lambda: C.ytime('raw_last_time', 'minute')),
      R(lambda: C.yp('raw_videos')), R(lambda: C.yp('raw_hours')))
    t('fewer than 50 new views (1,611 → 1,485 videos, 999,155 video-hours)', D('video filter of the converter', lambda: C.yp('processed_min_video_total')),
      R(lambda: C.yp('raw_videos')), R(lambda: C.yp('processed_videos')), R(lambda: C.yp('processed_rows')))
    t('collection gap on 19 May is never ground truth', D('gap day', lambda: C.ytime('gap_hours_list', 'day')))
    t('Excluding the 64 windows whose input contains it changes no mean by more than 0.0022',
      R(lambda: C.yp('common_windows_input64_contains_gap')), R(lambda: C.yp('gap_sensitivity_max_abs_diff')))
    # R4.11
    t('needed only in the first 32 windows of each data set', R(lambda: smax([Q(C.PS, 'padded_windows', scenario=sc, method='WSPI') for sc in M4])))
    t('4.9% of the evaluated windows on YouTube and 0.4%, 0.2% and 0.03% on the hourly, 30-minute and 5-minute taxi data',
      *[P(lambda sc=sc: Q(C.PS, 'padded_share', scenario=sc, method='WSPI')) for sc in M4], D('slot', C.slot('taxi_30min')), D('slot', C.slot('taxi_5min')))
    t('by at most 0.002 and its rank distortion by at most 0.2', R(lambda: C.pad_maxdiff(['ndcg@10_mean', 'rsi@10_mean'])), R(lambda: C.pad_maxdiff(['robustness_distortion_mean'])))
    t('edge extension was 0.0005 higher', R(lambda: Q(C.PD, 'ndcg@10_mean', scenario='taxi_5min', layer='ext', method='WSPI', mode='edge', subset='all')
                                                     - Q(C.PD, 'ndcg@10_mean', scenario='taxi_5min', layer='ext', method='WSPI', mode='symmetric', subset='all')))
    pdz = lambda sc: (Q(C.PD, 'ndcg@10_mean', scenario=sc, layer='ext', method='WSPI', mode='symmetric', subset='all')
                      - Q(C.PD, 'ndcg@10_mean', scenario=sc, layer='ext', method='DWT+AF', mode='zero', subset='all'))
    t('at an NDCG@10 that is 0.001 to 0.017 lower', R(lambda: smin([pdz(sc) for sc in M4])), R(lambda: smax([pdz(sc) for sc in M4])))
    t('Algorithm 1 (lines 1, 2 and 4)', X('algorithm number'), X('line of Algorithm 1'), X('line of Algorithm 1'), X('line of Algorithm 1'))
    # R4.12
    t('stays there for at least 6 consecutive slots, and was outside it for the 6 slots before', D('entry rule L', S(6, 'L')), D('entry rule P', S(6, 'P')))
    t('its median delay is 4 to 5 slots against 1 to 2 for AF, and it misses 16% to 31%',
      R(lambda: md('WSPI', smin)), R(lambda: md('WSPI', smax)), R(lambda: md('AF', smin)), R(lambda: md('AF', smax)),
      P(lambda: smin([C.resp(sc, 'WSPI', 'miss_rate') for sc in M4])), P(lambda: smax([C.resp(sc, 'WSPI', 'miss_rate') for sc in M4])))
    miss3 = lambda fn: fn([C.resp(sc, m, 'miss_rate', 'equal64') for sc in M4 for m in ['RRD', 'VSE', 'CompoundPop']])
    t('(RRD, VSE and CompoundPop) miss 33% to 55%', P(lambda: miss3(smin)), P(lambda: miss3(smax)))
    t('faster than RRD(64)', D('window length', w64))
    t('19.4% of the entries missed by WSPI are items with fewer than 32 observed slots',
      P(lambda: C.resp('taxi_5min', 'WSPI', 'share_miss_ineligible_whole_run')), D('observed slots required', lambda: C.cfgv('WSPI', 'min_obs')))
    rf = lambda col, fn: fn([Q(C.RF, col, scenario=sc, config='default', method='AF') for sc in M4])
    t('is −0.089 to 0.176, against 0.240 to 0.578 for AF',
      R(lambda: rf('spearman_ref_rsi_vs_truth_seen', smin)), R(lambda: rf('spearman_ref_rsi_vs_truth_seen', smax)),
      R(lambda: rf('spearman_rsi_vs_truth_seen', smin)), R(lambda: rf('spearman_rsi_vs_truth_seen', smax)))
    # R4.13
    t('the ground truth covers [t, t+1)', FORM())
    t('no YouTube video and 4 of 262 taxi zones were removed, the latter with at most 2 trips in any slot',
      R(lambda: Q(IU, 'items_removed', scenario='taxi_hourly')), R(lambda: Q(IU, 'items_in_file', scenario='taxi_hourly')),
      R(lambda: smax([Q(IU, 'removed_max_slot_count', scenario=sc) for sc in TAXI3])))
    t('no verdict of the 832 paired tests changed', R(lambda: S(len(verdicts_changed()[0]), f'{PT_W} and {PT_C}: paired rows (verdicts changed: {verdicts_changed()[1]})', (PT_W, PT_C))))
    t('on the first 25% of the data', D('truncation fraction of the audit', lambda: C.J(C.LJ, 'truncate_frac') * 100))
    t('(largest relative difference 3.2e-16)', R(lambda: S(C.load(C.TR).max_rel_diff.max(), f'{C.TR} max max_rel_diff', (C.TR,))))
    t('the 63 slots after it', D('perturbed slots of the audit minus the test slot', lambda: C.J(C.LJ, 'perturb_slots') - 1))
    # R4.14
    t('linear, μ_L(1 + R − W_E); product, μ_L·R·(1 − W_E)', FORM(), FORM())
    t('differs from that of the exponential form by less than 0.007', R(lambda: C.fusion_gap(RSI), op='lt'))
    t('its NDCG@10 differs by less than 0.005', R(lambda: C.fusion_gap(NDCG), op='lt'))
    t('when R = 0 or W_E = 1', FORM(), FORM())
    # R4.15
    t('(a) the default configuration (7 and 64 slots', D('baseline window length', w7), D('window length', w64))
    t('(37.12, 20.40, 12.60 and 7.03 on YouTube and the three taxi scenarios, 276.03 and 560.48 on the daily and weekly MovieLens data)',
      *[R(lambda sc=sc: V(sc, 'WSPI', DR)) for sc in M6])
    t('(0.9493 to 0.9897, against 0.8619 to 0.9446 for WSPI)',
      R(lambda: smin([V(sc, 'RRD', RSI, 'equal64') for sc in M6])), R(lambda: smax([V(sc, 'RRD', RSI, 'equal64') for sc in M6])),
      R(lambda: smin([V(sc, 'WSPI', RSI, 'equal64') for sc in M6])), R(lambda: smax([V(sc, 'WSPI', RSI, 'equal64') for sc in M6])))
    t('(for example NDCG@10 0.9061 against 0.8384 on the 5-minute taxi data)', R(lambda: V('taxi_5min', 'WSPI', NDCG, 'equal64')), R(lambda: V('taxi_5min', 'RRD', NDCG, 'equal64')),
      D('slot', C.slot('taxi_5min')))
    # R4.16
    t('(α = 2/(N+1))', FORM(), FORM())
    t("Holt's linear method (α = 0.3, γ = 0.1)", D('Holt alpha', lambda: C.J(EQ64_YT, 'holt', 'alpha')), D('Holt gamma', lambda: C.J(EQ64_YT, 'holt', 'gamma')))
    t('RSI@10 0.9897 against 0.9446 and NDCG@10 0.9651 against 0.9715 on YouTube, and on the 5-minute taxi data RSI@10 0.9686 against 0.9072 and NDCG@10 0.8379 against 0.9061',
      *[R(lambda m=m, met=met: V('youtube_hourly', m, met, 'equal64')) for met in (RSI, NDCG) for m in ('SMA', 'WSPI')], D('slot', C.slot('taxi_5min')),
      *[R(lambda m=m, met=met: V('taxi_5min', m, met, 'equal64')) for met in (RSI, NDCG) for m in ('SMA', 'WSPI')])
    # R4.18, software versions
    for i, l in enumerate(E.lines['L']):
        for m in re.finditer(r'(Python|NumPy|SciPy|pandas|PyWavelets|dtcwt|Matplotlib) (\d+\.\d+\.\d+)', l):
            for tk in E.toks:
                if tk.line == i + 1 and m.start(2) <= tk.col < m.end(2) and tk.status == 'unchecked':
                    exp = VERSIONS[m.group(1)]
                    tk.rule, tk.category, tk.source, tk.expected = 'software version', 'result', 'version recorded in the run metadata', exp
                    tk.status = 'ok' if exp == m.group(2) else 'mismatch'
# RULES_NUMBERS_END


# RULES_WORDS_BEGIN
LETTER = []          # the checked lines of the letter, set in main(); the structure checks below read them
MAIN_TEX = None
STRATA = 'T1.5_leakage_audit/causal/strata_usage.csv'
Res, over = W.Res, W.over


def _letter_line(anchor):
    h = [l for l in LETTER if anchor in l]
    if len(h) != 1:
        raise LookupError(f'letter anchor found {len(h)} times: {anchor!r}')
    return h[0]


def _items_after(anchor, stop, sep=r';\s*(?:and\s+)?|,\s*(?:and\s+)?|\s+and\s+'):
    l = _letter_line(anchor)
    seg = l.split(anchor, 1)[1].split(stop, 1)[0]
    return [x for x in re.split(sep, seg) if x.strip()]


@W.check('l_reviewers4', 'reviewers answered in the letter (headings "## Reviewer n")')
def _():
    n = sum(1 for l in LETTER if l.startswith('## Reviewer '))
    return Res(n == 4, f'{n} headings', 'letter')


@W.check('l_problems4', 'problems listed after "four problems that affect the reported numbers:" (reviewer version of ED.1)')
def _():
    it = _items_after('corrected four problems that affect the reported numbers: ', '. We also', sep=r';\s*(?:and\s+)?')
    return Res(len(it) == 4, f'{len(it)} items (split at ";")', 'letter')


@W.check('l_parts3', 'parts listed after "organised in three parts:" (R2.1) and bold labels of the Conclusion of the paper')
def _():
    it = _items_after('The Conclusion is now organised in three parts: ', '.')
    tex = MAIN_TEX.read_text(encoding='utf-8')
    con = tex.split(r'\section{Conclusion', 1)[1].split(r'\begin{thebibliography}', 1)[0]
    labels = re.findall(r'\\noindent\\textbf\{([^}]*)\}', con)
    ok = len(it) == 3 and len(labels) == 3
    return Res(ok, f'letter {len(it)}: {it}; paper {len(labels)}: {labels}', 'letter; paper tex (Conclusion)')


@W.check('l_sec31_paras', 'paragraphs of Section 3.1 in the paper tex (text before the notation table)')
def _():
    tex = MAIN_TEX.read_text(encoding='utf-8')
    sec = tex.split(r'\subsection{Formal Problem Definition}', 1)[1].split(r'\begin{table}', 1)[0]
    paras = [p for p in re.split(r'\n\s*\n', sec) if p.strip()]
    return Res(len(paras) == int(SEC31_PARAS), f'{len(paras)} paragraphs', 'paper tex')


@W.check('l_two_ops', 'R4.11 describes exactly two boundary operations, (i) and (ii)')
def _():
    l = _letter_line('Two boundary operations act on the wavelet coefficients')
    ok = '(i) Explicit padding' in l and '(ii) Boundary extension' in l and '(iii)' not in l
    return Res(ok, 'markers (i), (ii), no (iii)', 'letter')


@W.check('l_two_parts', 'R4.7 describes the shift test in two parts (First, Second)')
def _():
    l = _letter_line('We added a controlled shift test in two parts')
    ok = 'First, we shifted' in l and 'Second, we moved' in l and 'Third' not in l
    return Res(ok, 'First, Second, no Third', 'letter')


@W.check('l_two_examples', 'R3.4 gives two examples (PFRF; the 64-slot moving average)')
def _():
    it = _letter_line('Two examples from our results: ').split('Two examples from our results: ', 1)[1].split('. The results', 1)[0]
    n = len(it.split('; and '))
    return Res(n == 2, f'{n} examples (split at "; and")', 'letter')


@W.check('l_dwt_variants4', 'DWT variants of the ablation family (all compared with WSPI)')
def _():
    d = C.load(C.AB)
    v = sorted(d[(d.family == 'ablation') & d.variant.str.startswith('DWT')].variant.unique())
    return Res(len(v) == 4, f'{len(v)}: {v}', C.AB)


@W.check('l_dwt_32', 'block tests of RSI@10 and dRank, WSPI against the four DWT variants, four scenarios: all favour WSPI')
def _():
    d = C.load(C.ABT)
    x = d[d.method.str.startswith('DWT') & d.metric.isin([RSI, DR])]
    k = int((x.verdict == 'ref_better').sum())
    return Res(len(x) == 32 and k == 32, f'{k} of {len(x)} ref_better', C.ABT)


@W.check('l_dwtaf_zero_all4', 'extension modes: DWT+AF with zero extension has a higher RSI@10 than the default WSPI (each scenario)')
def _():
    g = lambda sc, m, mo: Q(C.PD, 'rsi@10_mean', scenario=sc, layer='ext', method=m, mode=mo, subset='all').value
    cases = [(sc, g(sc, 'DWT+AF', 'zero') > g(sc, 'WSPI', 'symmetric'), f"{g(sc, 'DWT+AF', 'zero'):.4f} vs {g(sc, 'WSPI', 'symmetric'):.4f}") for sc in M4]
    return over(cases, 'all', [C.PD])


@W.check('l_rho_top5', 'default: the highest rho of the nine is a wavelet-based method in five scenarios; AF on YouTube')
def _():
    cases = []
    for sc in M6:
        b, v = W.best('default', sc, RHO, ALL9)
        cases.append((W.SL[sc], b in W.WAV3, b))
    r = over(cases, 5, [C.VA])
    yb, _ = W.best('default', 'youtube_hourly', RHO, ALL9)
    r.ok = r.ok and yb == 'AF'
    return r


@W.check('l_rsi_ns_taxi', 'default, hourly and 30-minute taxi: DTCWT+AF has the highest RSI@10, not significantly above WSPI')
def _():
    cases = []
    for sc in ('taxi_hourly', 'taxi_30min'):
        b, v = W.best('default', sc, RSI, ALL9)
        vd = W.vd('default', sc, RSI, 'DTCWT+AF')
        cases.append((W.SL[sc], b == 'DTCWT+AF' and vd == 'n.s.', f'{b}, {vd}'))
    return over(cases, 'all', [C.VA, C.TA])


@W.check('l_rrd_eq', 'equal window, RRD vs WSPI, six scenarios: RSI@10 higher (6), dRank lower (4), NDCG@10 and rho lower (6); every verdict significant')
def _():
    out, ok = [], True
    for met, want_rrd_better, want in ((RSI, True, 6), (DR, True, 4), (NDCG, False, 6), (RHO, False, 6)):
        k = 0
        for sc in M6:
            vd = W.vd('equal64', sc, met, 'RRD')
            ok = ok and vd != 'n.s.'
            k += (vd == 'ref_worse') == want_rrd_better
        out.append(f'{met} {k}')
        ok = ok and k == want
    return Res(ok, '; '.join(out), C.TA)


@W.check('l_jsweep_mono', 'level sweep, N=64, WSPI and DTCWT+AF: RSI@10 rises and NDCG@10 falls with J (four scenarios)')
def _():
    d = C.load(LS)
    cases = []
    for m in ('WSPI', 'DTCWT+AF'):
        for sc in M4:
            x = d[(d.window == 64) & (d.method == m) & (d.scenario == sc)].sort_values('level')
            cases.append((f'{m} {W.SL[sc]}', x['rsi@10_mean'].is_monotonic_increasing and x['ndcg@10_mean'].is_monotonic_decreasing, 'monotone'))
    return over(cases, 'all', [LS])


@W.check('l_strata_never', 'STRATA_THRESHOLDS: defined only in config.py and on no import path of the V5 evaluation')
def _():
    d = C.load(STRATA)
    x = d[d.text.astype(str).str.contains('STRATA_THRESHOLDS')]
    files = sorted(set(x.file))
    ok = files == ['config.py'] and not x.on_v5_import_path.astype(str).eq('True').any()
    return Res(ok, f'files {files}', STRATA)


SEC31_PARAS = '5'


def rules_words(E):
    T = lambda anchor, *specs: E.text('L', anchor, list(specs))
    CL = lambda anchor, key, expect: E.claim('L', anchor, key, expect)
    Cl, Cn, Ds, Lg = W.Claim, W.Count, W.Design, W.Lang
    PL = lambda: Lg('ordinary wording, not a quantity')
    LOC = lambda: Lg('location in the paper, checked by reading it (section names are checked in part 1)')
    ALLR = lambda: Ds('statement about the whole revision: every result comes from the corrected protocol and every number of the paper and the letter is checked by check_paper_numbers.py, check_paper_word_numbers.py and this program')
    # words inside the quoted reviewer comments
    rx = re.compile(r'^\*\*(?:ED\.\d+|R\d\.\d+)\*\* \*(.*?)\*$')
    for i, l in enumerate(E.lines['L']):
        m = rx.match(l)
        if m:
            for tk in E.toks:
                if tk.line == i + 1 and m.start(1) <= tk.col < m.end(1) and tk.status == 'unchecked':
                    W.apply_spec(tk, Lg('reviewer comment (quoted)'), 'reviewer comment')
    # general response
    T('We thank the Editor and the four reviewers', Cn('l_reviewers4', '4'))
    T('checks every window of every run', Cl('audit_each_window', 'every window'), Cl('audit_every_run', 'every run'))
    T('the three wavelet-based methods used a 64-slot window. We now report two main tables', Cn('n_wav', '3'), Cn('n_configs', '2'))
    T('an equal 64-slot window for all nine methods. A full window-length sweep', PL(), Cn('n_methods', '9'))
    T('name the configuration under which each result holds', PL())
    T('All main results now include 95% confidence intervals', ALLR())
    T('a six-variant ablation', Cn('ablation_rows6', '6'))
    T('Below we answer each comment', PL())
    # ED.1 (full version)
    T('The observation window of every method was one slot shorter than stated', Ds('V4 audit, T1.1 (window_length_summary.csv): 63 and 6 slots'), Ds('V4 audit, T1.1'))
    T('part of the gap between the two methods', PL())
    T('filtered with a total count over the whole file, which uses future data', PL())
    T('In the robustness test each method chose its own target items and spike size; now all methods receive the same corrupted data',
      Ds('V4 test audited in T3.5 (current_test_audit.csv, R11 section 2)'), Ds('design of the V5 robustness test (common perturbation)'))
    # ED.1 (reviewer version)
    T('corrected four problems that affect the reported numbers: the observation windows were one slot shorter than stated',
      Cn('l_problems4', '4'), Ds('V4 audit, T1.1'))
    T('a count threshold over the whole file, which is now causal', PL())
    T('We also made the robustness test identical for all methods', Ds('design of the V5 robustness test (common perturbation)'))
    E.pattern(r'(All) results in the revised manuscript come from the corrected protocol, and (every) number was checked', ALLR(), group=1)
    E.pattern(r'All results in the revised manuscript come from the corrected protocol, and (every) number was checked', ALLR(), group=1)
    # ED.1, shared text
    T('the claims that WSPI is the most stable method in every setting', Lg('claim of the first version, quoted as withdrawn'), Lg('claim of the first version, quoted as withdrawn'))
    T('lowest rank displacement under spikes of all methods except PFRF in all six scenarios. Its RSI@10 is the highest',
      Cl('dr_low6', 'lowest of all except PFRF'), Cl('dr_low6', 'all 6'), Cn('n_scen6', '6'))
    CL('Its RSI@10 is the highest on YouTube, the 5-minute taxi data and the daily MovieLens data.', 'rsi_best_wspi3', 'WSPI highest on YouTube, taxi 5m, ML daily')
    CL('On the hourly and 30-minute taxi data DTCWT+AF is slightly higher, without a significant difference', 'l_rsi_ns_taxi', 'DTCWT+AF higher, n.s.')
    CL('on the weekly MovieLens data DTCWT+AF is significantly higher', 'rsi_best_wspi3', 'DTCWT+AF on ML weekly, significant')
    T('Section 4.3, results text and both main tables', Cn('n_configs', '2'))
    # ED.2, ED.3
    T('The exact code version used for all results', ALLR())
    T('All requested details are now in the manuscript', PL())
    T('Each point is listed below with its location', PL())
    T('locations are given under each item below', PL())
    # R1.1
    T('stated an application gap more than a methodological one', PL())
    T('in three training-free wavelet-based methods: DWT+AF', Cn('n_wav', '3'))
    # R1.2
    T('each covering eight slots, so μ_L tracks the trend at scales of about eight slots', PL(), Cn('eight_slots', '8'), Cn('eight_slots', '8'))
    T('for WSPI and DTCWT+AF, the four YouTube and taxi scenarios) confirms this in every scenario', Cn('n_scen4', '4'), Cl('l_jsweep_mono', 'every scenario'))
    T('We added a rationale and two experiments', Ds('the level sweep (T3.1) and the window sweep (T2.2)'))
    T('covers all methods (the wavelet-based methods from N = 16)', Ds('the sweep includes all nine methods (sweep_summary.csv)'))
    T('The window of every method is now listed in the configuration table', PL())
    # R1.3
    T('both structural terms get unit weight', Ds('the two structural terms R and W_E'))
    T('in the four YouTube and taxi scenarios, and a leakage-free check: the evaluation windows of each scenario', Cn('n_scen4', '4'), PL())
    T('and among these the one with the highest RSI@10 is taken', PL())
    T('within 1% of the best one', PL())
    T('change little over the whole grid', PL())
    T('their spread is at most 0.0104', Lg('"at most" before a checked number'))
    T('Section 4.9, both paragraphs and Table 13', LOC())
    # R1.4
    T('every method now sees the same corrupted data: in each window the same 50 low-activity items', Ds('design of the common perturbation (T3.5)'), PL())
    T('over five random seeds', Cn('seeds5', '5'))
    T('because the effect of a single spike falls', PL())
    T('and an equal 64-slot window for all methods, because', Cl('eq64_all', 'all nine methods'))
    T('with competitive ranking quality at every spike size', Cl('size_lowest', 'every size'))
    T('a six-slot burst', Cn('dur6', '6'))
    T('are computed over the whole window', PL())
    # R1.5
    T('On both granularities WSPI had the highest rank correlation', Cl('ml_rho_best', 'both'))
    T('(two MovieLens columns)', Cn('n_ml', '2'))
    # R1.6
    T('Consecutive windows share most of their input', PL())
    T('with a block of one day on YouTube (24 slots) and one week on the taxi data', Cn('block_day_week', '1 day'), Cn('block_day_week', '1 week'))
    T('autocorrelated over a week; a one-day block is reported', Ds('sensitivity block of one day (T1.6)'))
    T('All main results now come with 95% confidence intervals', ALLR())
    T('Each method is compared with WSPI', PL())
    T('with Holm correction within each scenario and metric; we base every claim on this block test', PL(), ALLR())
    T('the confidence intervals and all tests are in the Supplementary Information', PL())
    # R1.7
    T('measured runtime for all nine methods on one CPU core', PL(), Cn('n_methods', '9'), Cn('thread1', '1'))
    T('one million items take 12.5 s', Cn('million', '10^6'))
    T('scoring the whole catalogue of one window takes a median', PL(), PL())
    T('against slots of one hour and five minutes', Cn('slot_lengths', '1 h'), Cn('slot_lengths', '5 min'))
    # R1.8
    T('over all item-windows of the four YouTube and taxi scenarios', PL(), Cn('n_scen4', '4'))
    T('The two features are related by construction', Ds('the two structural features R and W_E'))
    T('they act together as one structural factor', PL())
    T('Section 3.5, last two sentences of the W_E paragraph', LOC())
    # R2.1
    T('The Conclusion is now organised in three parts', Cn('l_parts3', '3'))
    T('the evaluation covers three public datasets', Cn('n_datasets', '3'))
    T('recomputes the transform at each slot', PL())
    T('Conclusion in three parts (Findings', Cn('l_parts3', '3'))
    # R3.2, R3.3
    T('W_E was written in two ways', Ds('V4 text: W_E and WE (R18)'))
    T('j denoted both the imaginary unit and the level index', PL())
    T('the two real DWT trees', Ds('two trees of the DTCWT (theory, [27])'))
    T('the half-sample delay condition between their low-pass filters (new Eq. (1)), and why this makes the two wavelets',
      Ds('half-sample delay condition (theory, [27])'), Ds('two wavelets of the Hilbert pair (theory)'))
    T('with the half-sample delay condition as a new Eq. (1)', Ds('half-sample delay condition (theory, [27])'))
    # R3.4, R3.5
    T('Section 3.1 now separates the three notions and the metric used for each', Ds('three notions: quality, correlation, stability (Section 3.1)'), PL())
    T('Two examples from our results', Cn('l_two_examples', '2'))
    T('PFRF has the lowest rank displacement under spikes in every scenario', Cl('pfrf_lowest', 'every scenario'))
    T('paragraph on the three notions and metrics', Ds('three notions (Section 3.1)'))
    T('PFRF has the lowest ΔRank of all methods, but its Top-10 agrees poorly', Cl('pfrf_lowest', 'all methods'))
    T('tied among the top 21 items in nearly every window', Cl('pfrf_tied', 'nearly every window'))
    T('PFRF is kept in all tables', PL())
    # R3.6
    T('gives the largest change of all variants', Cl('dwt_largest_drop', 'largest'))
    T('and ΔRank more than doubles. All 32 comparisons', Cl('dwt_dr_double', 'more than doubles'), Cl('l_dwt_32', 'all 32'))
    T('between WSPI and the four DWT variants in the four YouTube and taxi scenarios favour WSPI', Cn('l_dwt_variants4', '4'), Cn('n_scen4', '4'))
    T('A similar gap appears between the two trend-only variants', Ds('the two trend-only variants (DTCWT and DWT) of the ablation'))
    T('in three of the four scenarios DWT gives a slightly higher NDCG@10', Cl('dwt_ndcg_3of4', '3'), Cn('n_scen4', '4'))
    # R3.7, R3.8, R3.9, R3.10
    T('The index itself uses the same settings on all datasets', Cl('config_same', 'all datasets'))
    T('(horizon of one slot, common windows', Cl('horizon_one', 'one slot'))
    T('Two settings depend on the dataset', Ds('the two dataset-dependent settings listed next: item threshold and bootstrap block'))
    T('The full configuration of every run is listed', PL())
    T('For each aspect (ranking quality', PL())
    T('what was shown on the three evaluated datasets', Cn('n_datasets', '3'))
    T('the results hold for these three public datasets', Cn('n_datasets', '3'))
    T('the findings with the configuration under which each holds', PL())
    T('The whole manuscript was proofread', PL())
    E.pattern(r'^\*\*Changes in the manuscript:\*\* (whole) manuscript', PL(), rule='location')
    # R4.1
    T('At ranking time t, each item i has an observation window', PL())
    T('The window length is stated for every method (N = 64 for the three wavelet-based methods', PL(), Cn('n_wav', '3'))
    T('Section 3.1 rewritten (five paragraphs)', Cn('l_sec31_paras', SEC31_PARAS))
    # R4.2
    T('in three training-free wavelet-based methods. DTCWT+AF and DWT+AF', Cn('n_wav', '3'))
    T('one of the three wavelet-based methods has the highest RSI@10 in all six evaluated scenarios (WSPI in three, DTCWT+AF in three) and the highest Spearman ρ in five of the six (AF on YouTube)',
      Cl('rsi_top_wav6', 'one wavelet-based method highest'), Cn('n_wav', '3'), Cl('rsi_top_wav6', 'all 6'), Cn('n_scen6', '6'),
      Cl('rsi_top_counts', 'WSPI 3'), Cl('rsi_top_counts', 'DTCWT+AF 3'), Cl('l_rho_top5', '5'), Cn('n_scen6', '6'))
    T('WSPI has the lowest rank displacement under spikes of all methods except PFRF in all six. The ablation',
      Cl('dr_low6', 'lowest of all except PFRF'), Cl('dr_low6', 'all 6'), Cn('n_scen6', '6'))
    T('now also positions the three methods against', Cn('n_wav', '3'))
    # R4.3
    T('**R4.3 Justification of the three features.**', Cn('features3', '3'))
    T('We now state what each feature measures', PL())
    T('We do not claim that three features are sufficient in general', Cn('features3', '3'))
    T('a scale term plus one bounded structural correction', PL())
    # R4.7
    T('We added a controlled shift test in two parts', Cn('l_two_parts', '2'))
    T('we shifted every real 64-slot window of every eligible item', Ds('design of part A of the shift test (T3.6)'), Ds('design of part A (T3.6)'))
    T('so its energy does not change at all', Cl('e1_zero', 'no change'))
    T('All these differences are significant in each of the four YouTube and taxi scenarios', Cl('shift_sig_all', 'all'), PL(), Cn('n_scen4', '4'))
    T('towards the newest slot, one slot at a time, and scored it with the settings of each method', Ds('step of the synthetic event (T3.6)'), PL())
    T('DTCWT+AF never falls', Cl('dtcwtaf_never', 'never'))
    T('the equivalent EWMA always rises', Cl('ewma_always', 'always'))
    T('it is one source among several', PL())
    # R4.8
    T('keeps the last N samples of each item', PL())
    T('recursive AF and EWMA a single value', PL())
    T('1 KiB in double precision', Ds('float64 coefficients (T3.8, state_memory.csv)'))
    # R4.9
    T('The ablation now has the six variants you suggested, all with the corrected protocol', Cn('ablation_rows6', '6'), ALLR())
    T('(the two other DWT variants are in the Supplementary Information)', Cn('ablation8', '8 - 6 = 2'))
    T('each term alone removes part of this gap, and the full index is the most robust', Cl('terms_each_part', 'each'), Cl('terms_each_part', 'most robust'))
    T('The two terms do not change RSI@10 significantly', Cl('terms_rsi', 'n.s. on three scenarios'))
    T('(by at most 0.0076). They lower NDCG@10 by at most 0.0042', Lg('"at most" before a checked number'), Lg('"at most" before a checked number'))
    T('Section 4.8, first two paragraphs and Table 12', LOC())
    # R4.10
    T('"YouTube videos viewCount every hour"', Lg('dataset title on Kaggle'))
    T('The ten-hour collection gap on 19 May is never ground truth', Cn('gap10', '10'), Cl('gap_never_truth', 'never'))
    T('an audit script that recomputes every number above and records the MD5 checksums of both files', Ds('tools/audit_youtube_provenance.py (T3.11)'), PL())
    # R4.11
    T('Two boundary operations act on the wavelet coefficients, and Section 3.4 (Stage 1) now describes both', Cn('l_two_ops', '2'), Cn('l_two_ops', '2'))
    T('in the first 32 windows of each data set', PL())
    T('changed NDCG@10 and RSI@10 of WSPI by at most 0.002 and its rank distortion by at most 0.2', Lg('"at most" before a checked number'), Lg('"at most" before a checked number'))
    T('The DTCWT filters extend the signal at both ends, at every level and in every window', Ds('boundary extension of the transform (dtcwt 0.14 colfilter; R10)'), Ds('R10'), Ds('R10'))
    T('We compared five extension modes', Cn('n_ext5', '5'))
    T('for the three wavelet-based methods in the four YouTube and taxi scenarios, with block-bootstrap', Cn('n_wav', '3'), Cn('n_scen4', '4'))
    T('gave the highest NDCG@10 in three scenarios', Cl('sym_ndcg3', '3'))
    T('but it had a higher rank distortion in every scenario', Cl('edge_dr_every', 'every'))
    T('in the single-slot spike test', PL())
    T('WSPI had the lowest rank distortion of the three wavelet-based methods under every mode', Cn('n_wav', '3'), Cl('wspi_min_mode', 'every mode'))
    T('its RSI@10 exceeds that of the default WSPI in all four scenarios', Cl('l_dwtaf_zero_all4', 'all'), Cn('n_scen4', '4'))
    # R4.12
    T('For each entry we measure', PL())
    T('AF misses none', Cl('af_none', 'none'))
    T('significantly faster than RRD(64) on the three taxi scenarios', Cn('n_taxi', '3'))
    T('WSPI has the largest rank displacement of all nine methods in the four YouTube and taxi scenarios', Cl('random_largest', 'largest of all'), Cn('n_methods', '9'), Cn('n_scen4', '4'))
    CL('WSPI is significantly faster than RRD(64) on the three taxi scenarios', 'long_slower', 'faster on taxi')
    # R4.13
    T('In every window the input covers', Cl('audit_each_window', 'every window'))
    T('the latter with at most 2 trips in any slot', Lg('"at most" before a checked number'))
    T('the two never overlap', Cl('horizon_one', 'input and truth never overlap'), Cl('horizon_one', 'never'))
    T('we found one use of future data in the first version', PL())
    T('filtered by the total count over the whole file', PL())
    T('All results were re-run with this rule', ALLR())
    T('checks every run: (i) the window structure of every window', Cl('audit_every_run', 'every run'), Cl('audit_each_window', 'every window'))
    T('is detected in every window', Cl('posctl_every', 'every window'))
    T('were never used in scoring or evaluation', Cl('l_strata_never', 'never'))
    # R4.14
    T('three alternatives built from the same features', Cn('fusion_alt3', '3'))
    T('The RSI@10 of each alternative differs', PL())
    T('The linear form is slightly more robust in every scenario', Cl('linear_more_robust', 'every scenario'))
    CL('the product form is more robust on the taxi data but less robust on YouTube', 'product_mixed', 'taxi yes, YouTube no')
    T('the exponential form is not the best in every metric', Lg('follows from the two sentences before it (checked claims)'))
    T('with weights above one', Ds('weights > 1 (property of the formula)'))
    T('one sentence after the fusion equation', LOC())
    # R4.15
    T('In the first version the six conventional baselines used a 7-slot window, and the three wavelet-based methods', Cn('n_base', '6'), Cn('n_wav', '3'))
    T('Each wavelet coefficient is built from all 64 slots', PL(), Ds('property of the transform (decision of 25 September)'))
    T('(b) an equal 64-slot window for all nine methods', PL(), Cn('n_methods', '9'))
    T('for all methods plus a simple moving average', Ds('the sweep includes all nine methods (sweep_summary.csv)'))
    T('In the default configuration, WSPI has the lowest rank displacement under spikes of all methods except PFRF in all six scenarios (37.12',
      Cl('dr_low6', 'lowest of all except PFRF'), Cl('dr_low6', 'all 6'), Cn('n_scen6', '6'))
    T('on YouTube and the three taxi scenarios, 276.03', Cn('n_taxi', '3'))
    T('MovieLens data), and all these differences are significant', Cl('dr_sig_all', 'all significant'))
    T('has a higher RSI@10 in all six scenarios (0.9493', Cl('l_rrd_eq', 'RSI@10 6'), Cn('n_scen6', '6'))
    T('and a lower rank displacement in four of them', Cl('l_rrd_eq', 'dRank 4'))
    T('WSPI has a higher NDCG@10 and Spearman ρ in all six scenarios (for example', Cl('l_rrd_eq', 'NDCG@10 and rho 6'), Cn('n_scen6', '6'))
    T('All these differences are significant. On the taxi data', Cl('l_rrd_eq', 'all significant'))
    T('The window of every method is listed in Table 6', PL())
    T('not as the most stable method', Lg('claim of the first version, quoted as withdrawn'))
    T('Section 4.3, two main tables and results text', Cn('n_configs', '2'))
    T('MovieLens added to both main tables', Cn('n_configs', '2'))
    # R4.16, R4.17, R4.18
    T("Holt's linear method (α = 0.3, γ = 0.1), each at N", PL())
    T('DWT+AF and DTCWT+AF remain in all tables', PL())
    T('one sentence on RRD and the moving average', LOC())
    T('Every real-time statement is now based on these measurements', ALLR())
    T('can rescore the whole catalogue at every slot on one core', PL(), Ds('the index is recomputed at each slot'), Cn('thread1', '1'))
    T('All figures and tables were regenerated', ALLR())
    T('the scripts that regenerate every table and figure', ALLR())
    T('on behalf of all authors', PL())
    # remaining plain wording
    E.pattern(r'under which (each) (?:result )?holds', PL(), rule='plain wording')
# RULES_WORDS_END


# PyWavelets 1.9.0 is the version installed for all V5 runs, but its generated
# pywt/version.py still says '1.8.0' (checked against the official cp312 win_amd64
# wheel of 1.9.0, git_revision c7bca20).  The run metadata therefore record 1.8.0;
# the paper, the SI and requirements.txt give the installed version.
PYWT_INSTALLED = {'1.8.0': '1.9.0'}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description='Check every number and number word of the response letter (T5.1)')
    ap.add_argument('--letter', required=True, type=Path, help='05_Response_to_Reviewers_DRAFT.md')
    ap.add_argument('--main', required=True, type=Path, help='WSPI_ScientificReports.tex of a built copy (the .aux is read next to it)')
    ap.add_argument('--results', required=True, type=Path, help='results/revision_v5')
    ap.add_argument('--figures-json', required=True, type=Path, help='Response/Figures/paper_print/export_paper_figures_run.json (Matplotlib version)')
    ap.add_argument('--out', required=True, type=Path, help='CSV report (one row per number and per word)')
    a = ap.parse_args()
    C.RESULTS = a.results
    global VERSIONS
    bench = C.load(C.BJ)['hardware']['versions']
    VERSIONS = {'Python': bench['python'], 'NumPy': bench['numpy'], 'pandas': bench['pandas'], 'SciPy': bench['scipy'],
                'PyWavelets': PYWT_INSTALLED.get(bench['pywt'], bench['pywt']), 'dtcwt': bench['dtcwt'],
                'Matplotlib': json.loads(a.figures_json.read_text(encoding='utf-8'))['matplotlib']}

    raw, lines = letter_lines(a.letter)
    MS.build('\n'.join(raw))                     # the reviewer version must build
    left = sorted(set(re.findall(r'\[\[[^\]]*\]\]', '\n'.join(lines))) - {'[[DOI]]'})

    E = LEngine(lines)
    labels, secs, eqs, bib = read_paper(a.main)
    rules_refs(E, labels, secs, eqs, bib)
    rules_quotes(E)
    rules_notation(E)
    rules_numbers(E)     # sentence rules first; the design patterns only fill what is left
    rules_design(E)

    global LETTER, MAIN_TEX
    LETTER, MAIN_TEX = lines, a.main
    W.E = W.Engine({'main': a.main})      # the structure checks of the paper read the paper through it
    WE = LWEngine(lines)
    rules_words(WE)

    rows = []
    for t in E.toks:
        rows.append(dict(kind='number', line=t.line, col=t.col, item=('-' if t.sign == '-' else '') + t.text, category=t.category or '-',
                         status=t.status, expected=t.expected, found='', check='', source=t.source, rule=t.rule, context=t.ctx.strip()))
    for t in WE.toks:
        rows.append(dict(kind='word', line=t.line, col=t.col, item=t.word, category=t.category or '-', status=t.status,
                         expected=t.expected, found=t.found, check=t.check, source=t.source, rule=t.rule, context=t.ctx.strip()))
    for x in WE.extra:
        rows.append(dict(kind='claim', line=x['line'], col=x['col'], item='', category='claim', status=x['status'], expected=x['expected'],
                         found=x['found'], check=x['check'], source=x['source'], rule=x['rule'], context=x['context']))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for kind in ('number', 'word', 'claim'):
        rs = [r for r in rows if r['kind'] == kind]
        c = Counter(r['status'] for r in rs)
        print(f'{kind}s: {len(rs)}  ' + '  '.join(f'{k}={v}' for k, v in sorted(c.items())))
    used = sorted({r['check'] for r in rows if r['check']})
    print(f'word checks used: {len(used)}')
    for p in left:
        print('PLACEHOLDER LEFT', p)
    problems = E.problems + WE.problems
    for p in problems:
        print('RULE PROBLEM', p)
    bad = [r for r in rows if r['status'] in ('mismatch', 'error', 'unchecked')]
    for r in bad:
        print(f"{r['status'].upper():9s} [letter:{r['line']}] {r['item'] or '(claim)':>10s} expected {r['expected']!s:>12s} | {r['rule'] or r['context'][:70]} | {r['found'] or r['source']}")
    n_bad = len(bad) + len(problems) + len(left)
    print('ALL NUMBERS OF THE LETTER AGREE WITH THE RESULT FILES AND THE PAPER' if n_bad == 0 else f'{n_bad} item(s) to review')
    return 0 if n_bad == 0 else 1


VERSIONS = {}

if __name__ == '__main__':
    sys.exit(main())
