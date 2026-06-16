#!/usr/bin/env python3
"""Combine everything into ONE corpus for a single brain that is pleasant to chat with AND
can translate goals into reward specs / commands.

  - reward frames   (language -> reward spec)      : keep ALL (consistent, repeated)
  - command frames  (act vs. ask-for-clarity)      : keep ALL (consistent, repeated)
  - modern dialogue (so it sounds normal, not a robot): SAMPLE to a byte budget so the
    high-variety chat doesn't drown out the consistent frames the small model relies on.

Conversation blocks (separated by blank lines) are shuffled together so training sees the
two skills interleaved. Output: data/mixed/chat.txt -> train with subdir 'mixed'.
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


def main(dialogue_budget=6_000_000, books_budget=60_000_000, seed=3):
    r = random.Random(seed)
    # instruction/skill frames — keep ALL (consistent, the behaviors we want)
    # NOTE: tooluse is intentionally EXCLUDED until speaking is genuinely good (user's gate)
    frames = (blocks("rewards") + blocks("commands") + blocks("chitchat")
              + blocks("misc") + blocks("dictionary") + blocks("rules"))
    # natural + broad everyday conversation (upweight convo so chat isn't shallow/off-topic)
    dialogue = (sample_to(r, blocks("dialogue_clean"), dialogue_budget)
                + blocks("convo") + blocks("convo"))
    books = sample_to(r, blocks("books"), books_budget)                  # clean language base
    allb = frames + dialogue + books
    r.shuffle(allb)
    out_dir = os.path.join(HERE, "data", "mixed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(allb) + "\n")
    print(f"[mixed] frames={len(frames):,}  dialogue={len(dialogue):,}  books={len(books):,} "
          f"-> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
