#!/usr/bin/env python3
"""CORPUS V3 — the rebalance the diagnostic called for. The v2/append recipe was ~43% DUPLICATED
templates and <9% conversation, which made the net a template-emitter that babbles in chat. V3:

  - CONVERSATION is the majority (real dialogue, upweighted) — chat is driven by chat, not frames.
  - books stay as a moderate raw-language base (the net completes these coherently).
  - instructional frames are kept at x1 (no drowning). Tool-use/reasoning are added in-kernel at
    LOW multipliers now (x2/x1, not x8/x5).
  - every block ends with an explicit EOS marker so the model learns where a reply STOPS.

Output: data/mixed_v3/chat.txt  (train with subdir 'mixed_v3')
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
EOS = " ■"          # ■  : single-token end-of-text marker (the regex tokenizes it as one)


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


def main(books_budget=90_000_000, dlg_modern_budget=22_000_000, seed=9):
    r = random.Random(seed)
    # CONVERSATION — the majority. Upweight the real dialogue we have (varied language, unlike
    # the rigid templates, so modest duplication is far less harmful).
    conversation = (blocks("dialogue_clean") * 3 + blocks("convo") * 6
                    + blocks("dialogue_classical") * 2 + blocks("chitchat") * 2
                    + sample_to(r, blocks("dialogue_modern"), dlg_modern_budget))
    # light instructional frames (x1)
    frames = (blocks("commands") + blocks("rules") + blocks("misc")
              + blocks("dictionary") + blocks("knowledge"))
    # raw-language base: books (moderate) + any modern/classical prose
    base = sample_to(r, blocks("books") + blocks("modern") + blocks("classical"), books_budget)

    allb = [b.strip() + EOS for b in (conversation + frames + base) if b.strip()]
    r.shuffle(allb)
    out_dir = os.path.join(HERE, "data", "mixed_v3")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(allb) + "\n")
    cb = sum(len(b) + 2 for b in conversation)
    fb = sum(len(b) + 2 for b in frames)
    bb = sum(len(b) + 2 for b in base)
    tot = cb + fb + bb
    print(f"[v3] conversation={cb/1e6:.0f}MB ({100*cb/tot:.0f}%)  frames={fb/1e6:.0f}MB "
          f"({100*fb/tot:.0f}%)  books={bb/1e6:.0f}MB ({100*bb/tot:.0f}%)  -> {path} "
          f"({os.path.getsize(path)/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
