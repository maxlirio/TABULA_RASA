#!/usr/bin/env python3
"""Build a clean, modern dictionary (word -> definition) from WordNet, restricted to common
words. Produces:
  - defs.json     : word -> gloss, for EXACT deterministic lookup at chat time (reliable;
                    the LM can't be trusted to recall the right definition).
  - data/dictionary/chat.txt : the same as USER/BOT frames, so the model also picks up
                    vocabulary during training.

WordNet is a public lexical database (not a pretrained language model) — using it keeps the
brain's knowledge grounded and exact, same principle as the rest of the memory.
"""
import json
import os

from nltk.corpus import wordnet as wn

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = {"the", "of", "and", "to", "in", "is", "it", "you", "that", "he", "was", "for",
        "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from",
        "or", "had", "by", "not", "but", "what", "all", "were", "a", "an"}


def best_def(w):
    """The definition of the most common sense of w (by WordNet lemma frequency)."""
    syns = wn.synsets(w)
    if not syns:
        return None
    best, best_count = syns[0], -1
    for s in syns:
        for lem in s.lemmas():
            if lem.name().lower() == w and lem.count() > best_count:
                best, best_count = s, lem.count()
    g = best.definition().strip()
    g = g.split(";")[0]                      # first clause only, keep it short
    return g[0].lower() + g[1:] if g else None


def main():
    common = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    out, glosses = [], {}
    for w in common:
        if w in STOP or not w.isalpha() or len(w) < 3:
            continue
        g = best_def(w)
        if not g or not (8 <= len(g) <= 120):
            continue
        glosses[w] = g
        out.append(f"USER: what does {w} mean?\nBOT: {w} means {g}.\n")
        out.append(f"USER: define {w}\nBOT: {g}.\n")
    out_dir = os.path.join(HERE, "data", "dictionary")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    json.dump(glosses, open(os.path.join(HERE, "defs.json"), "w"))
    print(f"[dictionary] {len(glosses):,} words -> defs.json + {len(out):,} training turns")


if __name__ == "__main__":
    main()
