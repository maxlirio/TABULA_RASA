#!/usr/bin/env python3
"""Pull a LARGE clean public-domain corpus from Project Gutenberg (via the Gutendex API),
downloading concurrently so it's fast. Strips boilerplate, writes data/books/chat.txt as
paragraph blocks — the language base for a bigger model.
"""
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 600
WORKERS = 16


def text_urls(target):
    urls, page = [], 1
    while len(urls) < target and page <= 80:
        try:
            req = urllib.request.Request(
                f"https://gutendex.com/books?languages=en&sort=popular&page={page}",
                headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(urllib.request.urlopen(req, timeout=20).read())
        except Exception:
            break
        for b in j.get("results", []):
            f = b.get("formats", {})
            u = (f.get("text/plain; charset=utf-8") or f.get("text/plain")
                 or f.get("text/plain; charset=us-ascii"))
            if u and not u.endswith(".zip"):
                urls.append(u)
        if not j.get("next"):
            break
        page += 1
    return urls[:target]


def fetch_clean(u):
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        s = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
    except Exception:
        return None
    if len(s) < 5000:
        return None
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", s, re.S)
    if m:
        s = s[m.end():]
    m = re.search(r"\*\*\* ?END OF", s)
    if m:
        s = s[:m.start()]
    s = (s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
         .replace("—", " - ").replace("–", "-").replace("\r", "")).lower()
    out = []
    for p in re.split(r"\n\s*\n", s):
        p = re.sub(r"\s+", " ", p).strip()
        if 60 <= len(p) <= 1200 and sum(c.isalpha() for c in p) > 0.6 * len(p):
            out.append(p)
    return out


def main():
    urls = text_urls(TARGET)
    print(f"[books] {len(urls)} URLs from Gutendex; downloading with {WORKERS} workers...",
          flush=True)
    paras, got = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(fetch_clean, urls):
            if res:
                got += 1
                paras.extend(res)
                if got % 25 == 0:
                    print(f"  {got} books, {len(paras):,} paragraphs", flush=True)
    out_dir = os.path.join(HERE, "data", "books")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras))
    print(f"[books] DONE {got} books -> {len(paras):,} paragraphs "
          f"({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)", flush=True)


if __name__ == "__main__":
    main()
