#!/usr/bin/env python3
"""Add FULL English Wikipedia (world knowledge + fluency) for the bigger model. Reads the per-article
JSON files of the 'Plain Text Wikipedia 2020-11' dataset (ltcmdrdata/plain-text-wikipedia-202011),
extracts article text, cleans it, and writes budget-capped paragraphs. Defensive about the exact JSON
shape (list-of-articles, JSONL, or dict) and finds the data under /kaggle/input or /content.

Output: data/bigwiki/chat.txt   (no EOS; t4_run adds it)
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_files():
    for root in ("/kaggle/input", "/content", os.getcwd()):
        # this dataset stores articles under enwiki*/*.json
        hits = sorted(glob.glob(f"{root}/**/enwiki*/*.json", recursive=True))
        if not hits:
            hits = [f for f in glob.glob(f"{root}/**/*.json", recursive=True) if "wiki" in f.lower()]
        if hits:
            return hits
    return []


def _articles(path):
    """Yield article text strings from one file, whatever its JSON shape."""
    try:
        d = json.load(open(path, encoding="utf-8", errors="ignore"))
        items = d if isinstance(d, list) else [d]
    except Exception:
        try:                                          # maybe JSONL (one object per line)
            items = [json.loads(l) for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
        except Exception:
            return
    for it in items:
        if isinstance(it, str):
            yield it
        elif isinstance(it, dict):
            t = it.get("text") or it.get("body") or it.get("content")
            if not t:                                 # fall back to the longest string field
                strs = [v for v in it.values() if isinstance(v, str)]
                t = max(strs, key=len) if strs else ""
            if t:
                yield t


def main(budget_mb=500):
    budget = int(budget_mb) * 1_000_000
    files = _find_files()
    if not files:
        print("[bigwiki] no wikipedia json found (attach ltcmdrdata/plain-text-wikipedia-202011) — skipping")
        return
    out, size = [], 0
    for f in files:
        for text in _articles(f):
            for para in re.sub(r"\r", "", text).split("\n"):
                para = re.sub(r"\s+", " ", para).strip()
                if len(para) >= 200 and para.count(".") >= 2:      # real paragraphs, not stubs/titles
                    out.append(para)
                    size += len(para)
            if size >= budget:
                break
        if size >= budget:
            break
    out_dir = os.path.join(HERE, "data", "bigwiki")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(out) + "\n")
    print(f"[bigwiki] {len(out):,} paragraphs, {size/1e6:.0f}MB from {len(files)} files "
          f"-> data/bigwiki/chat.txt", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
