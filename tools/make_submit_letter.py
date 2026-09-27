"""Build the reviewer version of the response letter from the full internal version.

Usage:
    python tools/make_submit_letter.py --src <05_Response_to_Reviewers_DRAFT.md> --out <05b_Response_to_Reviewers_SUBMIT.md>

Rules (T5.0, chat 19):
  * blocks between lines "[FULL-ONLY]" and "[/FULL-ONLY]" are dropped;
  * blocks between lines "[SUBMIT-ONLY]" and "[/SUBMIT-ONLY]" are kept, markers removed;
  * internal notes, i.e. paragraphs that start with "{" and end with "}", are dropped;
  * the task status after "Changes in the manuscript" (" — *applied (T4.1)*", " — *pending (T4.3)*") is dropped;
  * runs of blank lines are collapsed.
The script only reads --src and writes --out. It fails if a marker is unbalanced
or if an internal note or a marker survives in the output.
"""
import argparse
import re
import sys

STATUS = re.compile(r"\s+—\s+\*(?:applied|pending)[^*]*\*")


def build(text: str) -> str:
    out, mode = [], None
    for line in text.splitlines():
        tag = line.strip()
        if tag in ("[FULL-ONLY]", "[SUBMIT-ONLY]"):
            if mode is not None:
                raise ValueError(f"nested marker: {tag}")
            mode = tag
            continue
        if tag in ("[/FULL-ONLY]", "[/SUBMIT-ONLY]"):
            if mode is None or mode[1:] != tag[2:]:
                raise ValueError(f"unbalanced marker: {tag}")
            mode = None
            continue
        if mode == "[FULL-ONLY]":
            continue
        if tag.startswith("{") and tag.endswith("}"):
            continue
        out.append(STATUS.sub("", line))
    if mode is not None:
        raise ValueError(f"unclosed marker: {mode}")
    res = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
    for bad in ("[FULL-ONLY]", "[SUBMIT-ONLY]", "{Source", "*applied", "*pending"):
        if bad in res:
            raise ValueError(f"left in output: {bad}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        text = f.read()
    res = build(text)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(res)
    print(f"written {a.out}: {len(res.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
