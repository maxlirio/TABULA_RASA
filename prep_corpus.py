#!/usr/bin/env python3
"""Build a big CLEAN corpus from raw text sources, so the next run is properly FED (Chinchilla:
~20 tokens/param) instead of data-starved. Source-agnostic: point it at folders or files of
plain text and it strips boilerplate, normalizes, filters junk, DEDUPES paragraphs, and writes
up to a token budget.

  python3 prep_corpus.py <tokens_millions> <input_path> [<input_path> ...]
  e.g.  python3 prep_corpus.py 2000 ~/wiki_extract ~/openwebtext data/books

For SCALE without uploading gigabytes yourself: on Kaggle, ADD an existing hosted text dataset
(search "wikipedia", "openwebtext", "bookcorpus") as an input - it mounts under /kaggle/input -
and pass that path here. ~1 token ~= 4 characters, so 2,000M tokens ~= 8GB of text.
Output: data/web/chat.txt
"""
import glob
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_NORM = {"“": '"', "”": '"', "‘": "'", "’": "'",
         "—": " - ", "–": "-", "\r": ""}


def clean(para):
    for a, b in _NORM.items():
        para = para.replace(a, b)
    return re.sub(r"\s+", " ", para).strip().lower()


def strip_boilerplate(text):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)      # Project Gutenberg headers
    if m:
        text = text[m.end():]
    m = re.search(r"\*\*\* ?END OF", text)
    if m:
        text = text[:m.start()]
    return text


def iter_files(inputs):
    for inp in inputs:
        inp = os.path.expanduser(inp)
        if os.path.isdir(inp):
            yield from sorted(glob.glob(os.path.join(inp, "**", "*.txt"), recursive=True))
        elif os.path.isfile(inp):
            yield inp


def main(tokens_m=None, inputs=None):
    args = sys.argv[1:]
    if tokens_m is None:
        tokens_m = int(args[0]) if args and args[0].isdigit() else 1000
        inputs = args[1:] if len(args) > 1 else None
    inputs = inputs or ["data/books"]
    budget = int(tokens_m) * 1_000_000 * 4              # ~4 chars/token
    seen, kept, size, files = set(), [], 0, 0

    out_dir = os.path.join(HERE, "data", "web")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as fout:
        for fp in iter_files(inputs):
            if size >= budget:
                break
            try:
                text = open(fp, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            files += 1
            for para in re.split(r"\n\s*\n", strip_boilerplate(text)):
                p = clean(para)
                if not (60 <= len(p) <= 2000):
                    continue
                if sum(c.isalpha() for c in p) < 0.6 * len(p):   # drop tables/markup/junk
                    continue
                h = hashlib.md5(p.encode("utf-8")).hexdigest()   # dedupe identical paragraphs
                if h in seen:
                    continue
                seen.add(h)
                fout.write(p + "\n\n")
                size += len(p) + 2
                if size >= budget:
                    break
            if files % 200 == 0:
                print(f"  {files} files, {size/1e6:.0f}MB (~{size/4e6:.0f}M tokens)", flush=True)

    print(f"[corpus] {files} files -> {len(seen):,} unique paragraphs, {size/1e6:.0f}MB "
          f"(~{size/4e6:.0f}M tokens) -> {path}")
    print(f"[corpus] Chinchilla check: ~{size/4e6:.0f}M tokens trains a "
          f"~{size/4e6/20:.0f}M-param model well.")


if __name__ == "__main__":
    main()
