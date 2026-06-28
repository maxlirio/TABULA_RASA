#!/usr/bin/env python3
"""Clean WikiText-103 (wiki.train.tokens) into lowercased paragraphs (matching our corpus), up to
a byte budget, for MORE unique tokens. Strips WikiText markup (@-@ etc., = headings =). No EOS
marker — the corpus assembler adds it. Output: data/wiki/chat.txt
  python3 prep_wiki.py [budget_mb] [path_to_wiki.train.tokens]
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(budget_mb=250, path=None):
    src = path or next(iter(glob.glob("/kaggle/input/**/wiki.train.tokens", recursive=True)), None) \
        or next(iter(glob.glob("/kaggle/input/**/wiki.train.raw", recursive=True)), None)
    if not src or not os.path.exists(src):
        print("[wiki] wiki.train.tokens not found — attach the wikitext-103 dataset"); return
    budget = int(budget_mb) * 1_000_000
    out, size = [], 0
    with open(src, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("="):          # skip blanks and = headings =
                continue
            ln = ln.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")
            ln = re.sub(r"\s+", " ", ln).strip().lower()
            if len(ln) < 80:                          # keep real paragraphs, drop stubs
                continue
            out.append(ln)
            size += len(ln) + 2
            if size >= budget:
                break
    d = os.path.join(HERE, "data", "wiki")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "chat.txt"), "w", encoding="utf-8").write("\n\n".join(out) + "\n")
    print(f"[wiki] {len(out):,} paragraphs, {size/1e6:.0f}MB -> data/wiki/chat.txt")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250,
         sys.argv[2] if len(sys.argv) > 2 else None)
