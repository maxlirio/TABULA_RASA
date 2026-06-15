#!/usr/bin/env python3
"""Build a CONSISTENT-FRAME command corpus for a tiny model.

The idea (per the small-model rule of thumb): a sub-10M-param net learns a narrow skill far
better from a handful of grammatical frames repeated thousands of times with swapped-in
vocabulary than from noisy, every-line-different prose. So here we generate command/answer
turns from a small set of templates:

  - a clear command  -> the bot restates the intent (understanding, not just echoing)
  - a vague command  -> the bot asks ONE clarifying question and stays on topic

Output: data/commands/chat.txt in the same USER:/BOT: turn format the trainer already reads.
Nothing about the bot is hardcoded; this is training data it learns the pattern from.
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

VERBS = ["move", "push", "lift", "drop", "paint", "open", "close", "build", "remove",
         "copy", "rotate", "send", "carry", "turn", "raise", "lower", "find", "bring"]
NOUNS = ["box", "ball", "door", "light", "file", "block", "cup", "robot", "wheel", "arm",
         "table", "wall", "lever", "switch", "panel", "crate", "bottle", "key", "rope"]
ADJS = ["red", "blue", "green", "big", "small", "heavy", "light", "broken", "empty",
        "full", "left", "right", "round", "square", "metal", "wooden"]
PREPS = ["on", "under", "near", "beside", "behind", "inside", "above", "below"]
NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "x1", "y2", "z3", "core", "unit"]
NUMS = ["two", "three", "four", "five", "six", "seven"]


def frames(r):
    v, n, n2 = r.choice(VERBS), r.choice(NOUNS), r.choice(NOUNS)
    a, p, nm, num = r.choice(ADJS), r.choice(PREPS), r.choice(NAMES), r.choice(NUMS)
    # (user line, bot line) pairs — same frames, varied words. ~half clear, ~half need clarity.
    return [
        (f"{v} the {a} {n}", f"okay, i will {v} the {a} {n}."),
        (f"put the {n} {p} the {n2}", f"okay, i will put the {n} {p} the {n2}."),
        (f"make a {n} called {nm}", f"okay, i will make a {n} called {nm}."),
        (f"{v} {num} {n}s", f"okay, i will {v} {num} {n}s."),
        (f"if the {n} is {a} then {v} it", f"okay, when the {n} is {a} i will {v} it."),
        (f"can you {v} the {n}?", f"yes, i can {v} the {n}."),
        # vague -> ask one clarifying question, stay relevant
        (f"{v} it", f"what should i {v}?"),
        (f"{v} the {n}", f"which {n}? there may be more than one."),
        (f"{v} that one", f"which one do you mean?"),
        (f"do the thing with the {n}", f"what should i do with the {n}?"),
        (f"{v} them {p} there", f"{v} what, and where exactly?"),
    ]


def main(n_per_frame=1200, seed=1):
    r = random.Random(seed)
    out = []
    for _ in range(n_per_frame):
        for u, b in frames(r):
            out.append(f"USER: {u}\nBOT: {b}\n")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "commands")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"[commands] {len(out):,} turns -> {path} ({os.path.getsize(path):,} bytes)")


if __name__ == "__main__":
    main()
