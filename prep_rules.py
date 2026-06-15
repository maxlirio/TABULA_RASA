#!/usr/bin/env python3
"""Teach instruction-following FROM SCRATCH — several rule TYPES, and MULTIPLE rules at once.

The model only learns to "follow a rule as a rule" by seeing thousands of examples where a
rule is stated then obeyed, with the content randomized so it can't memorize. We cover:
  - replace / conditional : "say X instead of Y", "if i say Y say X", "answer Y with X"  -> Y->X
  - multi-rule            : 2-3 rules active at once, each input routed to the right one
  - echo                  : "repeat what i say" -> echoes the user's phrase (copy operation)
  - suffix                : "end every reply with X" -> appends X to normal replies

A slice of words is held out (never trained) so we can test generalization to UNSEEN rules.
Output: data/rules/chat.txt  (+ rules_holdout.json)
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

GREET = ["hello", "hi", "hey", "yes", "no", "thanks", "bye", "please", "sorry",
         "okay", "goodbye", "cool", "nice", "sure", "morning", "night"]
REPL_TMPL = [
    "say {x} instead of {y}", "use {x} instead of {y}", "replace {y} with {x}",
    "use {x} for {y}", "{y} now means {x}", "from now on {y} means {x}",
    "if i say {y} say {x}", "when i say {y} respond {x}", "answer {y} with {x}",
    "whenever i say {y} say {x}",
]
CHIT = [("how are you", "i'm good"), ("what is your name", "i'm apollo"),
        ("thank you", "you're welcome"), ("what time is it", "i don't know"),
        ("tell me a joke", "i'm still learning"), ("good morning", "good morning"),
        ("how is it going", "pretty good"), ("what's up", "not much"),
        ("are you there", "yes i'm here"), ("do you like me", "of course")]


def main(n_repl=9000, n_multi=6000, n_echo=2500, n_suffix=2500, seed=7):
    r = random.Random(seed)
    words = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    words = [w for w in words if w.isalpha() and 3 <= len(w) <= 10]
    train_words = words[:4000]
    holdout = words[4000:4350]                 # NEVER trained — for the generalization test
    pool = train_words + GREET
    out = []

    def apply_turns(mapping, n_apply, n_distract):
        turns = []
        for _ in range(n_apply):
            y = r.choice(list(mapping))
            turns.append((y, mapping[y]))
        for _ in range(n_distract):
            turns.append(r.choice(CHIT))
        r.shuffle(turns)
        return turns

    # single replacement / conditional
    for _ in range(n_repl):
        y, x = r.choice(GREET if r.random() < 0.5 else pool), r.choice(pool)
        if x == y:
            continue
        rule = r.choice(REPL_TMPL).format(x=x, y=y)
        turns = apply_turns({y: x}, r.randint(2, 3), r.randint(0, 2))
        out.append("\n".join([f"RULE: {rule}"] + [f"USER: {u}\nBOT: {a}" for u, a in turns]))

    # MULTIPLE rules at once (the routing fix)
    for _ in range(n_multi):
        k = r.randint(2, 3)
        ys = r.sample(pool, k)
        xs = [r.choice(pool) for _ in range(k)]
        mapping = {y: x for y, x in zip(ys, xs) if x != y}
        if len(mapping) < 2:
            continue
        rules = [r.choice(REPL_TMPL).format(x=mapping[y], y=y) for y in mapping]
        r.shuffle(rules)
        turns = apply_turns(mapping, r.randint(3, 5), r.randint(0, 2))
        out.append("\n".join([f"RULE: {rr}" for rr in rules]
                             + [f"USER: {u}\nBOT: {a}" for u, a in turns]))

    # echo / repeat (copy operation over arbitrary phrases)
    for _ in range(n_echo):
        rule = r.choice(["repeat what i say", "repeat after me", "echo what i say",
                         "say back what i say"])
        lines = [f"RULE: {rule}"]
        for _ in range(r.randint(2, 3)):
            phrase = " ".join(r.choice(pool) for _ in range(r.randint(1, 4)))
            lines.append(f"USER: {phrase}\nBOT: {phrase}")
        out.append("\n".join(lines))

    # suffix: end every reply with X
    for _ in range(n_suffix):
        x = r.choice(pool)
        rule = r.choice([f"end every reply with {x}", f"finish each answer with {x}",
                         f"always end with {x}"])
        turns = [r.choice(CHIT) for _ in range(r.randint(2, 3))]
        lines = [f"RULE: {rule}"] + [f"USER: {u}\nBOT: {a} {x}" for u, a in turns]
        out.append("\n".join(lines))

    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "rules")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    json.dump({"holdout": holdout}, open(os.path.join(HERE, "rules_holdout.json"), "w"))
    print(f"[rules] {len(out):,} rule-conversations (repl+multi+echo+suffix) -> data/rules/chat.txt; "
          f"{len(holdout)} held-out words")


if __name__ == "__main__":
    main()
