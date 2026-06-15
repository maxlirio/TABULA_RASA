#!/usr/bin/env python3
"""Assemble a big, clean, public-domain language corpus from Project Gutenberg, to give the
model real grammar/vocabulary breadth (the data bottleneck). Downloads a curated set of books
(narrative + simpler English), strips Gutenberg boilerplate, normalizes, and writes
data/books/chat.txt as paragraph blocks so it mixes into training as raw language.
"""
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# curated public-domain IDs: narrative novels + simpler/children's English + a few large ones
IDS = [11, 55, 16, 74, 76, 1342, 1661, 84, 345, 2701, 98, 1400, 1260, 768, 174, 158, 161,
       120, 35, 36, 5230, 43, 215, 271, 113, 146, 514, 236, 2591, 289, 45, 2781, 521, 829,
       219, 209, 33, 1184, 135, 2600, 996, 1232, 2554, 1727, 6130, 100, 730, 580, 766,
       1023, 1399, 25344, 28054, 64317, 2814, 28885, 408, 205, 203]


def fetch(i):
    for url in (f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt",
                f"https://www.gutenberg.org/files/{i}/{i}-0.txt",
                f"https://www.gutenberg.org/files/{i}/{i}.txt"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            s = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
            if len(s) > 5000:
                return s
        except Exception:
            continue
    return None


def clean(s):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", s, re.S)
    if m:
        s = s[m.end():]
    m = re.search(r"\*\*\* ?END OF", s)
    if m:
        s = s[:m.start()]
    s = (s.replace("“", '"').replace("”", '"').replace("‘", "'")
         .replace("’", "'").replace("—", " - ").replace("–", "-")
         .replace("\r", "")).lower()
    return s


def main():
    paras, got, chars = [], 0, 0
    for i in IDS:
        s = fetch(i)
        if not s:
            print(f"  (skip {i})")
            continue
        s = clean(s)
        got += 1
        chars += len(s)
        for p in re.split(r"\n\s*\n", s):
            p = re.sub(r"\s+", " ", p).strip()
            if 60 <= len(p) <= 1200 and sum(c.isalpha() for c in p) > 0.6 * len(p):
                paras.append(p)
        print(f"  got {i} ({len(s):,} chars, {got}/{len(IDS)})")
        time.sleep(0.5)
    out_dir = os.path.join(HERE, "data", "books")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras))
    print(f"[books] {got} books, {chars:,} raw chars -> {len(paras):,} paragraphs -> "
          f"data/books/chat.txt ({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
