#!/usr/bin/env python3
"""Pull a LARGE clean public-domain text corpus via the Gutendex API (reliable bulk access to
Project Gutenberg). Grabs the most popular English books, strips boilerplate, and writes
data/books/chat.txt as paragraph blocks. This is the language base for a bigger model.
"""
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 250          # how many books to collect


def text_urls(target):
    urls, page = [], 1
    while len(urls) < target and page <= 40:
        try:
            req = urllib.request.Request(
                f"https://gutendex.com/books?languages=en&sort=popular&page={page}",
                headers={"User-Agent": "Mozilla/5.0"})
            j = json.loads(urllib.request.urlopen(req, timeout=30).read())
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


def fetch(u):
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
    except Exception:
        return None


def clean(s):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", s, re.S)
    if m:
        s = s[m.end():]
    m = re.search(r"\*\*\* ?END OF", s)
    if m:
        s = s[:m.start()]
    return (s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
            .replace("—", " - ").replace("–", "-").replace("\r", "")).lower()


def main():
    urls = text_urls(TARGET)
    print(f"[books] {len(urls)} book URLs from Gutendex")
    paras, got, chars = [], 0, 0
    for i, u in enumerate(urls):
        s = fetch(u)
        if not s or len(s) < 5000:
            continue
        s = clean(s)
        got += 1
        chars += len(s)
        for p in re.split(r"\n\s*\n", s):
            p = re.sub(r"\s+", " ", p).strip()
            if 60 <= len(p) <= 1200 and sum(c.isalpha() for c in p) > 0.6 * len(p):
                paras.append(p)
        if got % 20 == 0:
            print(f"  {got} books, {chars:,} chars, {len(paras):,} paragraphs")
        time.sleep(0.3)
    out_dir = os.path.join(HERE, "data", "books")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras))
    print(f"[books] DONE {got} books, {chars:,} chars -> {len(paras):,} paragraphs "
          f"({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
