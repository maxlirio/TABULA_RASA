#!/usr/bin/env python3
"""Train the model to USE the logic engine as a TOOL, instead of us hand-coding a parser.

Each example teaches the full loop: turn natural language into a CALL the engine runs, then
turn the engine's RESULT into a natural reply. Inputs are phrased many ways so the model
learns to PARSE flexibly (the job the regex used to do, now learned). Every RESULT is computed
by the real engine (gm/know.py) so the data is always correct.

  USER: <natural language, many phrasings>
  CALL: store|find|ask|profile <args>
  RESULT: <what the engine returns>
  BOT: <natural reply generated from the result>

Output: data/tooluse/chat.txt
"""
import os
import random

from gm.know import Knowledge, sing
from gm.tools import build_reward
from prep_rewards import GOALS, REQ, DIRECT, VAGUE

HERE = os.path.dirname(os.path.abspath(__file__))

TEACH = {
    "isa": ["a {s} is a {o}", "a {s} is a kind of {o}", "{s}s are {o}s",
            "every {s} is a {o}", "did you know a {s} is a {o}", "i think a {s} is a {o}"],
    "has": ["a {s} has {o}", "a {s} has a {o}", "{s}s have {o}", "a {s} comes with {o}",
            "{s}s usually have {o}", "remember that a {s} has {o}"],
    "can": ["a {s} can {o}", "{s}s can {o}", "a {s} is able to {o}",
            "a {s} knows how to {o}", "{s}s are able to {o}"],
}
CONFIRM = ["got it.", "okay, noted.", "i'll remember that.", "understood.", "good to know.",
           "okay, i've stored that.", "noted."]
Q_FIND = ["what has {o}", "what things have {o}", "which things have {o}",
          "name something with {o}", "what do you know that has {o}"]
Q_ASK_HAS = ["does a {s} have {o}", "do {s}s have {o}", "has a {s} got {o}",
             "is it true that a {s} has {o}"]
Q_ASK_ISA = ["is a {s} a {o}", "are {s}s {o}s", "is a {s} a kind of {o}",
             "would you say a {s} is a {o}"]
Q_PROFILE = ["what is a {s}", "tell me about a {s}", "describe a {s}",
             "what do you know about {s}s", "what's a {s}"]


def vlist(xs):
    xs = [f"a {x}" for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def main(n=12000, reward_n=9000, seed=11):
    r = random.Random(seed)
    words = [w.strip().lower() for w in open(os.path.join(HERE, "common10k.txt"))]
    nouns = [w for w in words if w.isalpha() and 3 <= len(w) <= 9]
    verbs = ["run", "fly", "swim", "jump", "sing", "bark", "grow", "move", "shine",
             "float", "climb", "dig", "hunt", "sleep", "glow", "roll"]
    out = []

    for _ in range(n):
        k = Knowledge()
        cats = r.sample(nouns, 3)        # a small taxonomy: things -> category -> super
        thing, cat, sup = cats
        traits = r.sample(nouns, 2)
        act = r.choice(verbs)
        lines = []

        # teach facts via the tool
        def teach(s, rel, o):
            k._add(s, rel, o)
            nl = r.choice(TEACH[rel]).format(s=s, o=o)
            lines.extend([f"USER: {nl}", f"CALL: store {s} {rel} {o}",
                          "RESULT: ok", f"BOT: {r.choice(CONFIRM)}"])

        teach(thing, "isa", cat)
        teach(cat, "isa", sup)
        teach(cat, "has", traits[0])
        if r.random() < 0.6:
            teach(thing, "has", traits[1])
        if r.random() < 0.6:
            teach(cat, "can", act)

        # now ask the tool things (results computed by the real engine, incl. inheritance)
        qs = []
        o = traits[0]
        subs = k.subjects_with("has", o)
        qs.append((r.choice(Q_FIND).format(o=o), f"find has {o}",
                   ", ".join(subs) if subs else "none",
                   f"{vlist(subs)} {'has' if len(subs)==1 else 'have'} {o}." if subs
                   else f"nothing i know of has {o}."))

        yn = o in k.rel_objs(thing, "has")
        qs.append((r.choice(Q_ASK_HAS).format(s=thing, o=o), f"ask {thing} has {o}",
                   "yes" if yn else "no",
                   (f"yes, a {thing} has {o}." if yn else f"not that i know of.")))

        isayn = sup in k.ancestors(thing)
        qs.append((r.choice(Q_ASK_ISA).format(s=thing, o=sup), f"ask {thing} isa {sup}",
                   "yes" if isayn else "no",
                   (f"yes, a {thing} is a {sup}." if isayn else "not that i know of.")))

        parents = k.rel_objs(thing, "isa")
        prof = "isa " + ",".join(parents) if parents else "nothing"
        rep = (f"a {thing} is " + " and ".join(f"a {p}" for p in parents) + "."
               if parents else f"i don't know what a {thing} is.")
        qs.append((r.choice(Q_PROFILE).format(s=thing), f"profile {thing}", prof, rep))

        r.shuffle(qs)
        for nl, call, res, rep in qs[:r.randint(1, 3)]:
            lines.extend([f"USER: {nl}", f"CALL: {call}", f"RESULT: {res}", f"BOT: {rep}"])

        out.append("\n".join(lines))

    # ---- reward TOOL: route a goal to the reward builder, then verbalise the spec. The RESULT
    # is computed by the SAME build_reward() the runtime uses, so training and inference agree. ----
    goal_phrases = [g for g, _ in GOALS]
    for _ in range(reward_n):
        if r.random() < 0.78:
            g = r.choice(goal_phrases)
            user = r.choice(REQ).format(g=g)
            call = f"reward {g}"
        else:
            d = r.choice(list(DIRECT))
            verb = r.choice(["make it", "teach it to", "i want it to", "get it to"])
            user, call = f"{verb} {d}", f"reward {d}"
        spec = build_reward(call.split(" ", 1)[1])
        out.append(f"USER: {user}\nCALL: {call}\nRESULT: {spec}\nBOT: reward: {spec}")
    # vague reward requests -> ASK for the goal (no tool call), so it doesn't guess blindly
    for _ in range(max(1, reward_n // 20)):
        u, b = r.choice(VAGUE)
        out.append(f"USER: {u}\nBOT: {b}")

    r.shuffle(out)
    out_dir = os.path.join(HERE, "data", "tooluse")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chat.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(out) + "\n")
    print(f"[tooluse] {len(out):,} tool-use conversations -> data/tooluse/chat.txt "
          f"({os.path.getsize(os.path.join(out_dir,'chat.txt')):,} bytes)")


if __name__ == "__main__":
    main()
