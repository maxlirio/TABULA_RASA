#!/usr/bin/env python3
"""Teach the model to DESIGN rewards by understanding the goal — NOT keyword->template lookup.

Each example DERIVES the reward compositionally: the verb's ROLE applied to the goal's OBJECT,
with the REASONING shown. To make it GENERALIZE: (a) each role has MANY verb phrasings (so the
role is inferred from the verb, not memorized), and (b) a LARGE object pool (so the model learns
the object is a VARIABLE to plug in — "picking up X" -> +X_collected for ANY X, including kitchen/
potholes/phone it never saw). The reward is composed, never looked up; the model GENERATES it.
Output: data/reward_design/chat.txt
"""
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))

# (many gerund verbs, reasoning template, reward-term template). {o} = the goal's object (variable).
# CRITICAL: the object {o} must appear as its OWN standalone token in the spec, never glued to the
# sign/function (no "+{o}_collected" — that whole compound becomes ONE rare token, pruned by
# min_freq -> <unk>, so the model literally can't emit it). Instead we write "+collected {o}":
# the function word (+collected) is shared across thousands of examples (frequent, survives pruning)
# and {o} is a normal word the model already knows. This is genuinely compositional: emit the
# role-function, then plug in the object — works for ANY in-vocab object.
ROLES = [
    (["picking up", "collecting", "gathering", "clearing", "clearing away", "cleaning up",
      "tidying up", "removing", "scooping up", "sweeping up"],
     "less {o} around is the goal, so reward how much {o} gets collected and penalize {o} left behind",
     "+collected {o} -remaining {o} -time"),
    (["reducing", "minimizing", "lowering", "cutting down on", "decreasing", "limiting",
      "shrinking", "easing"],
     "the goal is LESS {o}, so penalize {o} directly while rewarding progress",
     "+progress -amount {o} -time"),
    (["increasing", "maximizing", "improving", "raising", "boosting", "growing", "building up"],
     "the goal is MORE {o}, so reward {o} directly and penalize waste",
     "+amount {o} -waste"),
    (["avoiding", "preventing", "steering clear of", "dodging", "staying away from",
      "keeping clear of", "not hitting"],
     "{o} is exactly what we don't want, so penalize {o} strongly while rewarding progress",
     "+progress -contact {o} -penalty"),
    (["reaching", "getting to", "navigating to", "arriving at", "finding", "going to",
      "making it to"],
     "the goal is to get to the {o}, so reward reaching it and penalize distance and time",
     "+reached {o} -distance -time"),
    (["cleaning", "keeping clean", "washing", "scrubbing", "wiping down", "keeping tidy",
      "maintaining", "keeping spotless"],
     "a clean {o} is the goal, so reward how clean it is and penalize mess left behind",
     "+cleanliness {o} -mess -time"),
    (["sorting", "organizing", "arranging", "ordering", "categorizing", "filing"],
     "order is the goal, so reward correct placement of {o} and penalize mistakes",
     "+sorted {o} -misplaced -time"),
    (["building", "constructing", "assembling", "putting together", "making"],
     "finishing the {o} is the goal, so reward progress and stability and penalize collapse",
     "+built {o} +stability -collapse"),
    (["protecting", "guarding", "defending", "keeping safe", "safeguarding", "securing"],
     "keeping the {o} safe is the goal, so reward it staying intact and penalize damage or loss",
     "+intact {o} -loss {o}"),
    (["learning", "memorizing", "mastering", "studying", "remembering"],
     "the goal is to know the {o}, so reward recall and penalize errors",
     "+recalled {o} -errors -time"),
    (["fixing", "repairing", "patching", "mending", "restoring"],
     "getting the {o} working is the goal, so reward it fixed and penalize remaining faults",
     "+fixed {o} -faults -time"),
    (["balancing", "stabilizing", "steadying", "keeping level", "keeping upright"],
     "keeping the {o} balanced is the goal, so reward staying level and penalize tipping",
     "+level {o} -tilt -fall"),
    (["charging", "filling", "topping up", "refueling"],
     "a full {o} is the goal, so reward the charge level and penalize overcharging or overheating",
     "+charge {o} -overcharge -overheat"),
    (["watering", "feeding", "growing", "tending"],
     "a healthy {o} is the goal, so reward its health/growth and penalize neglect",
     "+health {o} +growth -neglect"),
]
MODS = [
    (["safely", "without accidents", "carefully", "without anyone getting hurt"],
     "it also has to be safe", " -accidents -damage"),
    (["quickly", "fast", "as fast as possible", "without delay"], "speed matters too", " -time -delay"),
    (["efficiently", "using less energy", "without waste"], "it should be efficient", " -energy"),
    (["without breaking anything", "gently", "without damage"], "nothing should break", " -damage"),
]
REQ = ["design a reward for {g}", "make a reward for {g}", "how should i reward {g}",
       "design a reward system for {g}", "what's a good reward for {g}", "reward it for {g}",
       "give me a reward for {g}", "i need a reward for {g}", "what reward would you use for {g}",
       "set up a reward for {g}"]


def _objects():
    """A LARGE, varied object pool (common words) so the model treats the object as a variable."""
    words = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    pool = [w for w in words if w.isalpha() and 3 <= len(w) <= 10]
    extra = ["trash", "litter", "leaves", "weeds", "dust", "clutter", "dishes", "laundry", "toys",
             "snow", "crumbs", "spills", "noise", "waste", "errors", "potholes", "phone", "battery",
             "kitchen", "garden", "plant", "engine", "leak", "data", "files", "exit", "tower"]
    return list(dict.fromkeys(extra + pool))


def main(n=40000, seed=17):
    r = random.Random(seed)
    objs = _objects()
    out = []
    for _ in range(n):
        verbs, why, terms = r.choice(ROLES)
        o, verb = r.choice(objs), r.choice(verbs)
        goal = f"{verb} the {o}" if r.random() < 0.5 else f"{verb} {o}"
        reasoning, spec = why.format(o=o), terms.format(o=o)
        if r.random() < 0.45:
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
    print(f"[reward_design] {len(out):,} reasoned designs ({len(ROLES)} roles x {len(objs):,} "
          f"objects x mods) -> data/reward_design/chat.txt")


if __name__ == "__main__":
    main()
