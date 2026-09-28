r"""
Check the numbers of the Supplementary Information against the paper and the letter (task T4.10)
===============================================================================================
The paper cites SI items with fixed numbers ("Supplementary Table~S5"), because
the SI is a separate PDF (no xr package).  This program checks, after both files
have been compiled (the .aux files must exist):

  1. SI numbering: every SI label has the number agreed in R27 (EXPECTED_SI below);
  2. paper -> SI: every cited S-number exists in the SI; every SI table and figure is
     cited at least once; no 'S?' is left; the first citations run S1, S2, ... in
     order (numbering by first citation), separately for tables and figures;
  3. SI -> paper: every 'Table~N', 'Figure~N' and 'Section~N.M' written in the SI
     text exists in the paper, and the listed ones point at the expected label
     (EXPECTED_MAIN below);
  4. letter (optional): no '[[S?]]' or '[[yt_gap]]'/'[[padding]]' placeholder is left and
     every cited S-number exists in the SI.

Added for paper V5 (chat 28, 28 Sep 2026; decision of chat 27: fixed numbers plus a
check program).  Reads only; writes nothing unless --out is given (a CSV report).

Usage
-----
  python tools\check_si_refs.py --main <V5\source\WSPI_ScientificReports.tex>
         --si <V5\source\WSPI_SI.tex> [--letter <05_Response_to_Reviewers_DRAFT.md>] [--out report.csv]
Exit code 0 when every check passes, 1 otherwise.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

EXPECTED_SI = {   # numbers after task T4.12 (chat 31): MovieLens in the main results
    'tab:padding': 'S1', 'tab:si_features': 'S2', 'tab:si_levels': 'S3', 'tab:yt_gap': 'S4',
    'tab:si_ml_profile': 'S5', 'tab:si_ci_default': 'S6', 'tab:si_tests_default': 'S7',
    'tab:si_ci_equal64': 'S8', 'tab:si_tests_equal64': 'S9', 'tab:si_decomp': 'S10',
    'tab:si_robust_default': 'S11', 'tab:si_robust_equal64': 'S12', 'tab:si_hardware': 'S13',
    'tab:si_runtime_grid': 'S14', 'tab:si_memory': 'S15', 'tab:si_runtime_real': 'S16',
    'tab:si_ablation': 'S17', 'tab:si_fusion': 'S18', 'tab:si_selection': 'S19',
    'tab:si_resp_strict': 'S20', 'tab:si_rsi_truth': 'S21',
    'fig:si_window': 'S1', 'fig:si_features': 'S2', 'fig:si_ml_time': 'S3', 'fig:si_spike': 'S4',
    'fig:si_runtime': 'S5', 'fig:si_shift': 'S6', 'fig:si_grid': 'S7', 'fig:si_delay': 'S8',
}
# numbers of the paper written as text in the SI: (kind, number) -> label in the paper
EXPECTED_MAIN = {
    ('Table', '5'): 'tab:data', ('Table', '6'): 'tab:config', ('Table', '8'): 'tab:main_default',
    ('Table', '9'): 'tab:main_equal64', ('Table', '12'): 'tab:ablation', ('Table', '13'): 'tab:sens',
    ('Table', '14'): 'tab:resp',
}

LABEL = re.compile(r'\\newlabel\{((?:tab|fig|sec|subsec):[^}]*)\}\{\{([^}]*)\}')
# 'Supplementary Table~S5', 'Supplementary Tables~S21 and S22', 'Supplementary Fig.~S3 and Tables~S10 and S11'
SUPP = re.compile(r'Supplementary\s+((?:Fig\.|Figs\.|Figure|Figures|Table|Tables)~?\s*S[0-9?]+'
                  r'(?:(?:,|\s+and|\s+to|--)\s*(?:(?:Fig\.|Figs\.|Table|Tables)~?\s*)?S[0-9?]+)*)')
ITEM = re.compile(r'(Fig\.|Figs\.|Figure|Figures|Table|Tables)?~?\s*S([0-9?]+)')


def labels(aux):
    out = {}
    for m in LABEL.finditer(Path(aux).read_text(encoding='utf-8', errors='replace')):
        out[m.group(1)] = m.group(2)
    return out


def cited(text):
    """[(kind, number), ...] in order of appearance; kind = 'tab' or 'fig'."""
    res = []
    for m in SUPP.finditer(text):
        kind = None
        for it in ITEM.finditer(m.group(1)):
            if it.group(1):
                kind = 'fig' if it.group(1).startswith('Fig') else 'tab'
            res.append((kind, it.group(2)))
    return res


def main():
    ap = argparse.ArgumentParser(description='Check SI numbers and references')
    ap.add_argument('--main', required=True, type=Path)
    ap.add_argument('--si', required=True, type=Path)
    ap.add_argument('--letter', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    rows = []

    def check(what, ok, detail=''):
        rows.append(dict(check=what, passed=bool(ok), detail=detail))

    si_lab = labels(a.si.with_suffix('.aux'))
    main_lab = labels(a.main.with_suffix('.aux'))
    main_tex = a.main.read_text(encoding='utf-8')
    si_tex = a.si.read_text(encoding='utf-8')

    # 1. SI numbering
    for lab, num in EXPECTED_SI.items():
        check(f'SI label {lab} = {num}', si_lab.get(lab) == num, f'found {si_lab.get(lab)}')
    si_tab = {v for k, v in si_lab.items() if k.startswith('tab:')}
    si_fig = {v for k, v in si_lab.items() if k.startswith('fig:')}
    check('SI tables are S1..S21 without gaps', si_tab == {f'S{i}' for i in range(1, 22)}, str(sorted(si_tab)))
    check('SI figures are S1..S8 without gaps', si_fig == {f'S{i}' for i in range(1, 9)}, str(sorted(si_fig)))

    # 2. paper -> SI
    body = main_tex.split('\\begin{document}', 1)[1]
    check('no "S?" left in the paper', 'S?' not in body, f'{body.count("S?")} left')
    check('no "in the SI" left in the paper', ' in the SI' not in body)
    c = cited(body)
    for kind, n in c:
        if n == '?':
            continue
        pool = si_tab if kind == 'tab' else si_fig
        check(f'paper cites {kind} S{n}: exists in the SI', f'S{n}' in pool)
    for kind, pool in (('tab', si_tab), ('fig', si_fig)):
        seen = [f'S{n}' for k, n in c if k == kind and n != '?']
        miss = sorted(pool - set(seen), key=lambda x: int(x[1:]))
        check(f'every SI {kind} is cited in the paper', not miss, 'not cited: ' + ', '.join(miss))
        first = []
        for x in seen:
            if x not in first:
                first.append(x)
        check(f'first citations of SI {kind}s run in order', first == sorted(first, key=lambda x: int(x[1:])),
              ' '.join(first))

    # 3. SI -> paper
    si_body = si_tex.split('\\begin{document}', 1)[1]
    for d in sorted(Path(a.si.parent / 'si').glob('*.tex')):
        si_body += '\n' + d.read_text(encoding='utf-8')
    main_num = {}
    for k, v in main_lab.items():
        main_num.setdefault((k.split(':')[0], v), k)
    for m in re.finditer(r'(Tables?|Figures?)~(\d+)(?:\s+and~?(\d+))?', si_body):
        kind = 'Table' if m.group(1).startswith('Table') else 'Figure'
        for n in filter(None, (m.group(2), m.group(3))):
            key = ('tab' if kind == 'Table' else 'fig', n)
            lab = main_num.get(key)
            exp = EXPECTED_MAIN.get((kind, n))
            check(f'SI cites paper {kind} {n}', lab is not None and (exp is None or exp == lab),
                  f'paper label {lab}, expected {exp}')
    main_sec = {v for k, v in main_lab.items()}
    sec_aux = set(re.findall(r'\\contentsline\s*\{(?:sub)*section\}\{\\numberline\s*\{([0-9.]+)\}',
                             a.main.with_suffix('.aux').read_text(encoding='utf-8', errors='replace')))
    toc = a.main.with_suffix('.toc')
    if toc.exists():
        sec_aux |= set(re.findall(r'\\numberline\s*\{([0-9.]+)\}', toc.read_text(encoding='utf-8', errors='replace')))
    for m in re.finditer(r'Section~(\d+(?:\.\d+)?)', si_body):
        check(f'SI cites paper Section {m.group(1)}', m.group(1) in sec_aux or m.group(1) in main_sec,
              'sections found in the paper aux/toc' if sec_aux else 'no section list in the paper aux')

    # 4. letter
    if a.letter:
        lt = a.letter.read_text(encoding='utf-8')
        body = lt
        for ph in ('[[S?]]', '[[yt_gap]]', '[[padding]]'):
            check(f'no {ph} left in the letter', ph not in body, f'{body.count(ph)} left')
        for kind, n in cited(lt):
            if n == '?':
                continue
            pool = si_tab if kind == 'tab' else si_fig
            check(f'letter cites {kind} S{n}: exists in the SI', f'S{n}' in pool)

    bad = [r for r in rows if not r['passed']]
    for r in bad:
        print(f"[FAIL] {r['check']}  {r['detail']}")
    print(f'{len(rows) - len(bad)} of {len(rows)} checks passed')
    if a.out:
        with open(a.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['check', 'passed', 'detail'])
            w.writeheader()
            w.writerows(rows)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
