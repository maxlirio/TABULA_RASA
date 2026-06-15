#!/usr/bin/env python3
"""Turn DailyDialog (clean, everyday human conversation) into USER:/BOT: turns. This is far
cleaner than the movie-dialogue corpus (no slang/shouting/non-sequiturs), so the model learns
more natural, fluid back-and-forth. Output: data/dialogue_clean/chat.txt
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "dd_tmp", "EMNLP_dataset", "dialogues_text.txt")


def clean(u):
    u = (u.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")).lower()
    u = re.sub(r"(\w)\s*'\s*(\w)", r"\1'\2", u)   # "don ' t" -> "don't"
    u = re.sub(r"\s+([,.!?;:])", r"\1", u)        # "stinks ." -> "stinks."
    u = re.sub(r"\s+", " ", u)
    return u.strip()


def main():
    out, kept = [], 0
    for line in open(SRC, encoding="utf-8", errors="ignore"):
        utts = [clean(u) for u in line.split("__eou__")]
        utts = [u for u in utts if 1 <= len(u) <= 160]
        if len(utts) < 2:
            continue
        kept += 1
        for i, u in enumerate(utts):
            out.append(("USER: " if i % 2 == 0 else "BOT: ") + u)
        out.append("")
    out_dir = os.path.join(HERE, "data", "dialogue_clean")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[dialogue_clean] {kept:,} conversations -> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
