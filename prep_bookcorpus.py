#!/usr/bin/env python3
"""Add BookCorpus (narrative prose / dialogue) for the bigger model — the register it lacks and the
one that matters for storytelling + problem-solving reasoning. Streams the big CSV in chunks (never
loads it whole), auto-detects the text column, and joins consecutive rows into paragraph-sized
passages (BookCorpus is often one sentence per row). Budget-capped. Finds the CSV under
/kaggle/input or /content (nishantsingh96/refined-bookcorpus-dataset).

Output: data/bookcorpus/chat.txt   (no EOS; t4_run adds it)
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_csv():
    for root in ("/kaggle/input", "/content", os.getcwd()):
        hits = [f for f in glob.glob(f"{root}/**/*.csv", recursive=True) if "book" in f.lower()]
        if hits:
            return max(hits, key=os.path.getsize)     # the biggest book*.csv
    return None


def main(budget_mb=500):
    budget = int(budget_mb) * 1_000_000
    csv = _find_csv()
    if not csv:
        print("[bookcorpus] no book csv found (attach nishantsingh96/refined-bookcorpus-dataset) — skipping")
        return
    import pandas as pd
    out, size, buf, text_col = [], 0, [], None

    def flush():
        para = re.sub(r"\s+", " ", " ".join(buf)).strip()
        buf.clear()
        if len(para) >= 200 and para.count(".") >= 2:
            out.append(para)
            return len(para)
        return 0

    for chunk in pd.read_csv(csv, chunksize=20000, dtype=str, on_bad_lines="skip",
                             engine="python", quoting=3):
        if text_col is None:                          # the column with the longest average strings
            text_col = max(chunk.columns, key=lambda c: chunk[c].astype(str).str.len().mean())
        for cell in chunk[text_col].dropna():
            buf.append(str(cell).strip())
            if sum(len(b) for b in buf) >= 400:       # accumulate ~paragraph-sized passages
                size += flush()
        if size >= budget:
            break
    size += flush()
    out_dir = os.path.join(HERE, "data", "bookcorpus")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(out) + "\n")
    print(f"[bookcorpus] {len(out):,} passages, {size/1e6:.0f}MB (text col {text_col!r}) "
          f"-> data/bookcorpus/chat.txt", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
