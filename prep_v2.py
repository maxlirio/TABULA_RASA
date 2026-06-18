#!/usr/bin/env python3
"""Assemble CORPUS V2 — rebalanced to fine-tune ON TOP of the current model (warm-start) and
work on its ROUGH AREAS, rather than repeat the last run's book-heavy mix:

  - rough area: garbled / archaic "book-bleed" chat  -> upweight CLEAN + MODERN conversation,
    and shrink the classic-book share (it was ~90% of the last corpus and dominated training).
  - rough area: following rules & commands           -> upweight the rules + commands frames.
  - rough area: WHEN to use tools                    -> tool-use data is added in-kernel (with
    the don't-call contrast set) on top of this.

A book base is kept (enough to preserve vocabulary), but the TRAINING SIGNAL now leans on the
weak skills instead of prose. Output: data/mixed_v2/chat.txt  (train with subdir 'mixed_v2').
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def blocks(subdir):
    path = os.path.join(HERE, "data", subdir, "chat.txt")
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8", errors="ignore").read()
    return [b.strip() for b in text.split("\n\n") if b.strip()]


def sample_to(r, items, budget):
    r.shuffle(items)
    kept, size = [], 0
    for b in items:
        if size > budget:
            break
        kept.append(b)
        size += len(b) + 2
    return kept


def main(books_budget=170_000_000, dlg_modern_budget=14_000_000, seed=5):
    r = random.Random(seed)
    # SKILL / INSTRUCTION frames — upweighted (these are the weak behaviors we want to drill):
    # rules x3 and commands x3 so "follow this instruction" gets real training signal.
    frames = (blocks("commands") * 3 + blocks("rules") * 3 + blocks("chitchat") * 2
              + blocks("misc") + blocks("dictionary") + blocks("knowledge"))
    # CLEAN, MODERN conversation upweighted hard (fixes garbled/archaic replies):
    dialogue = (blocks("dialogue_clean") * 2 + blocks("convo") * 4
                + sample_to(r, blocks("dialogue_modern"), dlg_modern_budget))
    # classic-book base kept only for VOCABULARY, not to dominate (warm-start already gave language)
    books = sample_to(r, blocks("books"), books_budget)
    allb = frames + dialogue + books
    r.shuffle(allb)
    out_dir = os.path.join(HERE, "data", "mixed_v2")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(allb) + "\n")
    fb = sum(len(b) + 2 for b in frames)
    db = sum(len(b) + 2 for b in dialogue)
    bb = sum(len(b) + 2 for b in books)
    tot = fb + db + bb
    print(f"[v2] frames={fb/1e6:.1f}MB  dialogue={db/1e6:.1f}MB  books={bb/1e6:.1f}MB  "
          f"-> {path} ({os.path.getsize(path)/1e6:.0f}MB)")
    print(f"[v2] skill+chat share = {(fb+db)/tot:.0%} (last corpus was ~10%)")


if __name__ == "__main__":
    main()
