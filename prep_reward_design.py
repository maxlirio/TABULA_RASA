#!/usr/bin/env python3
"""Teach the model to DESIGN rewards by understanding the goal — NOT keyword->template lookup.

Each example DERIVES the reward compositionally: the verb's ROLE (pick-up -> collect & penalize
what's left; reduce -> minus; avoid -> penalize; reach -> reward arrival; ...) applied to the
goal's OBJECT, with the REASONING shown. Roles are paired with sensible objects so the goals are
real, but the reward is COMPOSED (role x object), never looked up — across thousands of combos the
model learns the OPERATION, so it generalizes to unseen goals ("picking up trash" -> less trash is
good). The model GENERATES the reward (with its reasoning), not a tool. Output: data/reward_design/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (gerund verbs, sensible object pool, reasoning about the object's role, reward template). {o}=object
ROLES = [
    (["picking up", "collecting", "gathering", "clearing away", "cleaning up"],
     ["trash", "litter", "leaves", "toys", "crumbs", "scraps", "cans", "bottles", "sticks", "mail"],
     "less {o} around is the goal, so reward how much {o} gets collected and penalize any {o} left behind",
     "+{o}_collected -{o}_remaining -time"),
    (["reducing", "minimizing", "lowering", "cutting down on"],
     ["waste", "noise", "cost", "delay", "risk", "clutter", "pollution", "downtime"],
     "the goal is LESS {o}, so penalize {o} directly while rewarding progress",
     "+progress -{o} -time"),
    (["increasing", "maximizing", "improving", "raising"],
     ["output", "score", "savings", "accuracy", "coverage", "yield", "uptime", "throughput"],
     "the goal is MORE {o}, so reward {o} directly and penalize waste",
     "+{o} -waste"),
    (["avoiding", "preventing"],
     ["errors", "accidents", "collisions", "mistakes", "leaks", "spills", "crashes", "falls"],
     "{o} is exactly what we don't want, so penalize it strongly",
     "+progress -{o}(strong)"),
    (["reaching", "getting to", "navigating to", "finding"],
     ["exit", "target", "goal", "door", "base", "top", "finish", "checkpoint"],
     "the goal is to get to the {o}, so reward reaching it and penalize distance and time",
     "+reached_{o} -distance -time"),
    (["cleaning", "washing", "scrubbing", "tidying"],
     ["room", "floor", "table", "kitchen", "window", "dishes", "car", "yard"],
     "a clean {o} is the goal, so reward cleanliness and penalize mess left",
     "+{o}_cleaned -mess_left -time"),
    (["sorting", "organizing", "arranging"],
     ["books", "papers", "files", "tools", "boxes", "cards", "mail", "parts"],
     "order is the goal, so reward correct placement and penalize mistakes",
     "+{o}_in_order -misplaced -time"),
    (["building", "constructing", "assembling"],
     ["tower", "bridge", "wall", "model", "robot", "house", "shelf", "track"],
     "finishing the {o} is the goal, so reward progress and stability and penalize collapse",
     "+{o}_built +stability -collapse"),
    (["protecting", "guarding", "defending"],
     ["data", "files", "base", "cargo", "goods", "vault", "perimeter"],
     "keeping the {o} safe is the goal, so reward it staying intact and penalize damage or loss",
     "+{o}_intact -{o}_loss"),
    (["learning", "memorizing", "mastering"],
     ["words", "facts", "names", "route", "song", "lines", "recipe", "moves"],
     "the goal is to know the {o}, so reward recall and penalize errors",
     "+{o}_recalled -errors -time"),
    (["fixing", "repairing", "patching"],
     ["engine", "leak", "bug", "pipe", "wire", "door", "roof", "circuit"],
     "getting the {o} working is the goal, so reward it fixed and penalize remaining faults",
     "+{o}_fixed -faults -time"),
    (["balancing", "stabilizing", "steadying"],
     ["load", "tray", "stack", "pole", "boat", "platform", "ladder"],
     "keeping the {o} balanced is the goal, so reward staying level and penalize tipping",
     "+level -tilt -fall"),
]
MODS = [
    (["safely", "without accidents", "carefully"], "it also has to be safe", " -accidents -damage"),
    (["quickly", "fast"], "speed matters here too", " -time(strong)"),
    (["efficiently", "using less energy"], "it should also be efficient", " -energy"),
    (["without breaking anything", "gently"], "nothing should get broken", " -damage"),
]
REQ = ["design a reward for {g}", "make a reward for {g}", "how should i reward {g}",
       "design a reward system for {g}", "what's a good reward for {g}", "reward it for {g}",
       "give me a reward for {g}", "i need a reward for {g}"]


def main(n=22000, seed=17):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        verbs, objs, why, terms = r.choice(ROLES)
        o, verb = r.choice(objs), r.choice(verbs)
        goal = f"{verb} the {o}" if r.random() < 0.55 else f"{verb} {o}"
        reasoning, spec = why.format(o=o), terms.format(o=o)
        if r.random() < 0.5:
            mwords, mwhy, mterms = r.choice(MODS)
            goal = f"{goal} {r.choice(mwords)}"
            reasoning += f", and since {mwhy}, penalize the bad case"
            spec += mterms
        out.append(f"USER: {r.choice(REQ).format(g=goal)}\nBOT: {reasoning}. reward: {spec}")
    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "reward_design")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[reward_design] {len(out):,} reasoned, composed reward designs "
          f"({len(ROLES)} roles, sensible objects, +mods) -> data/reward_design/chat.txt")


if __name__ == "__main__":
    main()
