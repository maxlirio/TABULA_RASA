#!/usr/bin/env python3
"""Long-reply / STORYTELLING data: so the model learns that SOME replies are long, and stops the
canned 'i can't tell stories' dodge. Pairs story/longer-answer requests with multi-sentence
paragraphs sourced from the books (real narrative prose). Output: data/stories/chat.txt

Honest note: this teaches the model to ATTEMPT longer replies; coherence over a full paragraph is
still capped by model scale (it will drift), but it stops the reflexive one-line deflection.
"""
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = ["tell me a story", "tell me a story", "can you tell me a story", "tell me a little story",
       "i'd love to hear a story", "tell me something", "say something longer", "spin me a tale",
       "tell me a tale", "read me something", "tell me a bedtime story", "give me a longer answer",
       "talk to me for a bit", "tell me a story please"]


def main(n=6000, seed=21):
    r = random.Random(seed)
    src = os.path.join(HERE, "data", "books", "chat.txt")
    if not os.path.exists(src):
        print("[stories] no data/books — skipping"); return
    text = open(src, encoding="utf-8", errors="ignore").read()
    paras = []
    for p in text.split("\n\n"):
        p = re.sub(r"\s+", " ", p).strip()
        if 140 <= len(p) <= 520 and p.count(".") + p.count("!") + p.count("?") >= 2:
            paras.append(p)
    if not paras:
        print("[stories] no suitable paragraphs"); return
    r.shuffle(paras)
    out = []
    for i in range(min(n, len(paras))):
        out.append(f"USER: {r.choice(REQ)}\nBOT: {paras[i]}")
    out_dir = os.path.join(HERE, "data", "stories")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[stories] {len(out):,} long-reply story turns (from {len(paras):,} book paragraphs) "
          f"-> data/stories/chat.txt")


if __name__ == "__main__":
    main()
