#!/usr/bin/env python3
"""Teach the SHAPE of thinking: step-by-step reasoning where every chain is COMPUTED by the
real logic engine / calculator, so the reasoning shown is always correct (not plausible-but-
wrong). The model learns to decompose a problem and ground each step, instead of mimicking
reasoning it can't actually do. Three kinds:

  - deduction       : transitive is-a chains      ("a robin is a bird, a bird is an animal -> yes")
  - inheritance     : property inheritance         ("birds have feathers, a robin is a bird -> yes")
  - word problems   : decomposition -> CALL calc   (the arithmetic is done by the tool)

Output: data/reasoning/chat.txt
"""
import os
import random

from gm.know import Knowledge
from gm.tools import calc

HERE = os.path.dirname(os.path.abspath(__file__))
THINK = ["let's think step by step.", "let's reason it out.", "let me work through it.",
         "step by step:", "let's think it through.", "okay, reasoning it out:"]


def art(w):
    return ("an " if w[:1] in "aeiou" else "a ") + w


def plural(w):
    return w if w.endswith("s") else w + "s"


def main(n=9000, seed=23):
    r = random.Random(seed)
    words = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    nouns = [w for w in words if w.isalpha() and 3 <= len(w) <= 9]
    containers = ["boxes", "shelves", "bags", "baskets", "rows", "crates", "jars", "bins"]
    out = []

    for _ in range(n):
        kind = r.random()

        if kind < 0.38:                                   # transitive is-a deduction (yes or no)
            a, b, c, d = r.sample(nouns, 4)
            k = Knowledge()
            k._add(a, "isa", b)
            k._add(b, "isa", c)
            if r.random() < 0.7:                          # provable YES
                q = f"all {plural(a)} are {plural(b)}. all {plural(b)} are {plural(c)}. is {art(a)} {c}?"
                assert c in k.ancestors(a)
                ans = (f"{art(a)} is {art(b)}, and {art(b)} is {art(c)}, "
                       f"so yes - {art(a)} is {art(c)}.")
            else:                                         # NOT derivable -> honest NO
                q = f"all {plural(a)} are {plural(b)}. all {plural(b)} are {plural(c)}. is {art(a)} {d}?"
                assert d not in k.ancestors(a)
                ans = (f"{art(a)} is {art(b)}, and {art(b)} is {art(c)}, but nothing tells me "
                       f"{art(c)} is {art(d)}, so no - not that i can tell.")
            out.append(f"USER: {q}\nBOT: {r.choice(THINK)} {ans}")

        elif kind < 0.7:                                  # property inheritance
            a, b = r.sample(nouns, 2)
            trait = r.choice(nouns)
            k = Knowledge()
            k._add(b, "has", trait)
            k._add(a, "isa", b)
            assert trait in k.rel_objs(a, "has")
            q = f"{plural(b)} have {trait}. {art(a)} is {art(b)}. does {art(a)} have {trait}?"
            ans = f"{art(a)} is {art(b)}, and {plural(b)} have {trait}, so yes - {art(a)} has {trait}."
            out.append(f"USER: {q}\nBOT: {r.choice(THINK)} {ans}")

        else:                                             # word problem -> decompose to the tool
            x, y = r.randint(2, 12), r.randint(2, 12)
            thing = r.choice(nouns)
            cont = r.choice(containers)
            res = calc(f"{x}*{y}")
            q = f"there are {x} {cont} with {y} {plural(thing)} in each. how many {plural(thing)} in total?"
            out.append(f"USER: {q}\nBOT: {r.choice(THINK)} that's {x} groups of {y}, so {x} "
                       f"times {y}.\nCALL: calc {x}*{y}\nRESULT: {res}\n"
                       f"BOT: so there are {res} {plural(thing)} in total.")

    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "reasoning")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[reasoning] {len(out):,} grounded reasoning chains -> data/reasoning/chat.txt "
          f"({os.path.getsize(os.path.join(out_dir, 'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
