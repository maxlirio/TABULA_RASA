#!/usr/bin/env python3
"""Test whether it learned to follow a rule AS A RULE — i.e. it applies a substitution rule
to word pairs it was NEVER trained on. That's the proof of genuine generalization (the rule),
not memorization (the words)."""
import json
import os
import random
import sys

from gm.chat import Chat
from gm.lm import load

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "apollo.pt"


def fresh(voices):
    c = Chat(voices, "apollo", os.path.join(HERE, "_rt.json"))
    c.bot_name = c.user_name = c.persona = None
    c.notes, c.history, c.session, c.rules = [], [], [], []
    c.know.triples = []
    c.asked = set()
    return c


def main():
    model, coder = load(os.path.join(HERE, CKPT))
    voices = {"apollo": (model, coder)}
    hold = json.load(open(os.path.join(HERE, "rules_holdout.json")))["holdout"]
    invocab = [w for w in hold if w in coder.stoi]      # must be representable
    r = random.Random(1)
    pairs = []
    while len(pairs) < 10 and invocab:
        x, y = r.choice(invocab), r.choice(invocab)
        if x != y:
            pairs.append((x, y))

    print(f"Held-out words usable: {len(invocab)}.  Testing {len(pairs)} NEVER-SEEN rule pairs:\n")
    hits = 0
    for x, y in pairs:
        c = fresh(voices)
        c.reply(f"say {x} instead of {y}")
        out = c.reply(y).strip().lower()
        ok = out == x
        hits += ok
        print(f"  rule: say '{x}' instead of '{y}'   | you:'{y}' -> bot:'{out}'   {'HIT' if ok else 'miss'}")

    print(f"\nGeneralization: {hits}/{len(pairs)} held-out rules applied correctly.")

    # control: with NO rule, it should NOT substitute
    print("\nControl (no rule set):")
    for y in [p[1] for p in pairs[:3]]:
        c = fresh(voices)
        print(f"  you:'{y}' -> bot:'{c.reply(y).strip()}'  (should NOT be a forced substitution)")

    # honest OOV check: a made-up word like 'florp' is out-of-vocabulary for a word model
    print("\nOOV check ('florp' is not a real word -> not in vocab):",
          "in vocab" if "florp" in coder.stoi else "NOT in vocab (expected)")
    for p in (os.path.join(HERE, "_rt.json"),):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
